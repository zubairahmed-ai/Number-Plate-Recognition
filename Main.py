# Main.py

import cv2
import numpy as np
import os
import time
import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

def main(image):

    CnnClassifier = DetectChars.loadCNNClassifier()         # attempt KNN training
    #response  = str(input('Do you want to see the Intermediate images: '))
    """
    if response == 'Y' or response == 'y':
        showSteps = True
    else:
        showSteps = False

    """

    if CnnClassifier == False:                               # if KNN training was not successful
        print("\nerror: CNN traning was not successful\n")               # show error message
        return                                                          # and exit program

    imgOriginalScene  = cv2.imread(image)               # open image
    h, w = imgOriginalScene.shape[:2]
    # As the image may be blurr so we sharpen the image.
    #kernel_shapening4 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    #imgOriginalScene = cv2.filter2D(imgOriginalScene,-1,kernel_shapening4)
    
    #imgOriginalScene = cv2.resize(imgOriginalScene,(1000,600),interpolation = cv2.INTER_LINEAR)
    
    imgOriginalScene = cv2.resize(imgOriginalScene, (0, 0), fx = 1.4, fy = 1.4,interpolation=cv2.INTER_LINEAR)
    
    #imgOriginalScene = cv2.fastNlMeansDenoisingColored(imgOriginalScene,None,10,10,7,21)
    
    #imgOriginal = imgOriginalScene.copy()
    
    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")      # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates. We get a list of
                                                                                        # combinations of contours that may be a plate.


    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    if showSteps == True:
        cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")             # inform user no plates were found
        response = ' '
        return response,imgOriginalScene
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        if showSteps == True:
            cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
            cv2.waitKey(0)
        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")       # show message
            return ' ',imgOriginalScene                                       # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        print("\nlicense plate read from ", image," :",licPlate.strChars,"\n")
        print("----------------------------------------")

        if showSteps == True:
            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

            cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file
            cv2.waitKey(0)                    # hold windows open until user presses a key

    return licPlate.strChars,licPlate.imgPlate
###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect. Here, bounding rectangle is drawn with minimum area, so it considers the rotation also

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(dir_path + '/Test_images/538E945.jpg')
    main(dir_path + '/Test_images/BKF196.jpg')
    main(dir_path + '/Test_images/489T051.jpg')

#Output
# "C:\Users\Zubair Ahmed\AppData\Local\Programs\Python\Python35\python.exe" "C:/Users/Zubair Ahmed/Downloads/Number-Plate-Recognition-master/Number-Plate-Recognition-master/Main.py"
# Using TensorFlow backend.
# 2018-02-18 21:46:56.094382: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 1/1 [==============================] - 0s
# 3
# 1/1 [==============================] - 0s
# 8
# 1/1 [==============================] - 0s
# E
# 1/1 [==============================] - 0s
# 9
# 1/1 [==============================] - 0s
# 4
# 1/1 [==============================] - 0s
# 5
# 1/1 [==============================] - 0s
# 6
# 1/1 [==============================] - 0s
# D
# 1/1 [==============================] - 0s
# 3
# 
# license plate read from  
# /Test_images/538E945.jpg  : 38E945 
# 
# ----------------------------------------
# 1/1 [==============================] - 0s
# B
# 1/1 [==============================] - 0s
# K
# 1/1 [==============================] - 0s
# F
# 1/1 [==============================] - 0s
# 1
# 1/1 [==============================] - 0s
# 9
# 1/1 [==============================] - 0s
# 6
# 
# license plate read from  
# /Test_images/BKF196.jpg  : BKF196 
# 
# ----------------------------------------
# 1/1 [==============================] - 0s
# 4
# 1/1 [==============================] - 0s
# 8
# 1/1 [==============================] - 0s
# 9
# 1/1 [==============================] - 0s
# T
# 1/1 [==============================] - 0s
# 0
# 1/1 [==============================] - 0s
# 5
# 1/1 [==============================] - 0s
# 7
# 
# license plate read from  
# /Test_images/489T051.jpg  : 489T057 
# 
# ----------------------------------------
# 
# Process finished with exit code 0


