from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _get_scale(self):
    p1 = Vector(*self.to_parent(0, 0))
    p2 = Vector(*self.to_parent(1, 0))
    scale = p1.distance(p2)
    if hasattr(self, '_scale_p'):
        if str(scale) == str(self._scale_p):
            return self._scale_p
    self._scale_p = scale
    return scale