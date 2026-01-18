from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def update_bbox(self, touch):
    """Update gesture bbox from a touch coordinate"""
    x, y = (touch.x, touch.y)
    bb = self.bbox
    if x < bb['minx']:
        bb['minx'] = x
    if y < bb['miny']:
        bb['miny'] = y
    if x > bb['maxx']:
        bb['maxx'] = x
    if y > bb['maxy']:
        bb['maxy'] = y
    self.width = bb['maxx'] - bb['minx']
    self.height = bb['maxy'] - bb['miny']
    self._update_time = Clock.get_time()