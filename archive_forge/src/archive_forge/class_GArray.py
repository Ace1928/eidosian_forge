import sys
import cv2 as cv
@register('cv2')
class GArray:

    def __new__(cls, argtype):
        return cv.GArrayT(argtype)

    class Bool:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_BOOL)

    class Int:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_INT)

    class Int64:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_INT64)

    class UInt64:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_UINT64)

    class Double:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_DOUBLE)

    class Float:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_FLOAT)

    class String:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_STRING)

    class Point:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_POINT)

    class Point2f:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_POINT2F)

    class Point3f:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_POINT3F)

    class Size:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_SIZE)

    class Rect:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_RECT)

    class Scalar:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_SCALAR)

    class Mat:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_MAT)

    class GMat:

        def __new__(self):
            return cv.GArrayT(cv.gapi.CV_GMAT)

    class Prim:

        def __new__(self):
            return cv.GArray(cv.gapi.CV_DRAW_PRIM)

    class Any:

        def __new__(self):
            return cv.GArray(cv.gapi.CV_ANY)