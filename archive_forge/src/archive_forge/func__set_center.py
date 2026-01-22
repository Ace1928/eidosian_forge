from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _set_center(self, center):
    if center == self.center:
        return False
    t = Vector(*center) - self.center
    trans = Matrix().translate(t.x, t.y, 0)
    self.apply_transform(trans)