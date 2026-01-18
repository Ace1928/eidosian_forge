from kivy.animation import Animation
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import (ObjectProperty, StringProperty,
from kivy.uix.widget import Widget
from kivy.logger import Logger
def toggle_layout(*l):
    o = acc.orientation
    acc.orientation = 'vertical' if o == 'horizontal' else 'horizontal'