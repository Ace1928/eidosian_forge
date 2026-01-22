import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
class GesturePoint:

    def __init__(self, x, y):
        """Stores the x,y coordinates of a point in the gesture."""
        self.x = float(x)
        self.y = float(y)

    def scale(self, factor):
        """ Scales the point by the given factor."""
        self.x *= factor
        self.y *= factor
        return self

    def __repr__(self):
        return 'Mouse_point: %f,%f' % (self.x, self.y)