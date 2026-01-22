import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def _center_gesture(self):
    """ Centers the Gesture.points of the gesture."""
    total_x = 0.0
    total_y = 0.0
    total_points = 0
    for stroke in self.strokes:
        stroke_y = sum([pt.y for pt in stroke.points])
        stroke_x = sum([pt.x for pt in stroke.points])
        total_y += stroke_y
        total_x += stroke_x
        total_points += len(stroke.points)
    if total_points == 0:
        return False
    total_x /= total_points
    total_y /= total_points
    for stroke in self.strokes:
        stroke.center_stroke(total_x, total_y)
    return True