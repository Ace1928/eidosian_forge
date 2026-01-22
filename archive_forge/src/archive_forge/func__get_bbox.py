from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _get_bbox(self):
    xmin, ymin = xmax, ymax = self.to_parent(0, 0)
    for point in [(self.width, 0), (0, self.height), self.size]:
        x, y = self.to_parent(*point)
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y
    return ((xmin, ymin), (xmax - xmin, ymax - ymin))