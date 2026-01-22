from time import time
from kivy.event import EventDispatcher
from kivy.properties import NumericProperty, BooleanProperty
from kivy.clock import Clock
def apply_distance(self, distance):
    if abs(distance) < self.min_distance:
        self.velocity = 0
    self.value += distance