import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def _show_handles(self, dt):
    if not self.use_handles or not self.text:
        return
    win = EventLoop.window
    handle_right = self._handle_right
    handle_left = self._handle_left
    if self._handle_left is None:
        self._handle_left = handle_left = Selector(source=self.handle_image_left, target=self, window=win, size_hint=(None, None), size=('45dp', '45dp'))
        handle_left.bind(on_press=self._handle_pressed, on_touch_move=self._handle_move, on_release=self._handle_released)
        self._handle_right = handle_right = Selector(source=self.handle_image_right, target=self, window=win, size_hint=(None, None), size=('45dp', '45dp'))
        handle_right.bind(on_press=self._handle_pressed, on_touch_move=self._handle_move, on_release=self._handle_released)
    else:
        if self._handle_left.parent:
            self._position_handles()
            return
        if not self.parent:
            return
    self._trigger_position_handles()
    if self.selection_from != self.selection_to:
        self._handle_left.opacity = self._handle_right.opacity = 0
        win.add_widget(self._handle_left, canvas='after')
        win.add_widget(self._handle_right, canvas='after')
        anim = Animation(opacity=1, d=0.4)
        anim.start(self._handle_right)
        anim.start(self._handle_left)