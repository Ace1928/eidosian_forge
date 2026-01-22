from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@property
def animated_properties(self):
    return ChainMap({}, self.anim2.animated_properties, self.anim1.animated_properties)