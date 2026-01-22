import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
class GestureStroke:
    """ Gestures can be made up of multiple strokes."""

    def __init__(self):
        """ A stroke in the gesture."""
        self.points = list()
        self.screenpoints = list()

    @property
    def max_x(self):
        if len(self.points) == 0:
            return 0
        return max(self.points, key=lambda pt: pt.x).x

    @property
    def min_x(self):
        if len(self.points) == 0:
            return 0
        return min(self.points, key=lambda pt: pt.x).x

    @property
    def max_y(self):
        if len(self.points) == 0:
            return 0
        return max(self.points, key=lambda pt: pt.y).y

    @property
    def min_y(self):
        if len(self.points) == 0:
            return 0
        return min(self.points, key=lambda pt: pt.y).y

    def add_point(self, x, y):
        """
        add_point(x=x_pos, y=y_pos)
        Adds a point to the stroke.
        """
        self.points.append(GesturePoint(x, y))
        self.screenpoints.append((x, y))

    def scale_stroke(self, scale_factor):
        """
        scale_stroke(scale_factor=float)
        Scales the stroke down by scale_factor.
        """
        self.points = [pt.scale(scale_factor) for pt in self.points]

    def points_distance(self, point1, point2):
        """
        points_distance(point1=GesturePoint, point2=GesturePoint)
        Returns the distance between two GesturePoints.
        """
        x = point1.x - point2.x
        y = point1.y - point2.y
        return math.sqrt(x * x + y * y)

    def stroke_length(self, point_list=None):
        """Finds the length of the stroke. If a point list is given,
           finds the length of that list.
        """
        if point_list is None:
            point_list = self.points
        gesture_length = 0.0
        if len(point_list) <= 1:
            return gesture_length
        for i in range(len(point_list) - 1):
            gesture_length += self.points_distance(point_list[i], point_list[i + 1])
        return gesture_length

    def normalize_stroke(self, sample_points=32):
        """Normalizes strokes so that every stroke has a standard number of
           points. Returns True if stroke is normalized, False if it can't be
           normalized. sample_points controls the resolution of the stroke.
        """
        if len(self.points) <= 1 or self.stroke_length(self.points) == 0.0:
            return False
        target_stroke_size = self.stroke_length(self.points) / float(sample_points)
        new_points = list()
        new_points.append(self.points[0])
        prev = self.points[0]
        src_distance = 0.0
        dst_distance = target_stroke_size
        for curr in self.points[1:]:
            d = self.points_distance(prev, curr)
            if d > 0:
                prev = curr
                src_distance = src_distance + d
                while dst_distance < src_distance:
                    x_dir = curr.x - prev.x
                    y_dir = curr.y - prev.y
                    ratio = (src_distance - dst_distance) / d
                    to_x = x_dir * ratio + prev.x
                    to_y = y_dir * ratio + prev.y
                    new_points.append(GesturePoint(to_x, to_y))
                    dst_distance = self.stroke_length(self.points) / float(sample_points) * len(new_points)
        if not len(new_points) == sample_points:
            raise ValueError('Invalid number of strokes points; got %d while it should be %d' % (len(new_points), sample_points))
        self.points = new_points
        return True

    def center_stroke(self, offset_x, offset_y):
        """Centers the stroke by offsetting the points."""
        for point in self.points:
            point.x -= offset_x
            point.y -= offset_y