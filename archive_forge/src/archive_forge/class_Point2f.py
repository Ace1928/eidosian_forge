import sys
import cv2 as cv
class Point2f:

    def __new__(self):
        return cv.GArrayT(cv.gapi.CV_POINT2F)