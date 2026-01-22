from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _set_do_translation(self, value):
    if type(value) in (list, tuple):
        self.do_translation_x, self.do_translation_y = value
    else:
        self.do_translation_x = self.do_translation_y = bool(value)