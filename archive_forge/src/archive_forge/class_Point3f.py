import sys
import cv2 as cv
class Point3f:

    def __new__(self):
        return cv.GArrayT(cv.gapi.CV_POINT3F)