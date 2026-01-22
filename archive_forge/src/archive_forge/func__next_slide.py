from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def _next_slide(self):
    if len(self.slides) < 2:
        return None
    if self.loop and self.index == len(self.slides) - 1:
        return self.slides[0]
    if self.index < len(self.slides) - 1:
        return self.slides[self.index + 1]