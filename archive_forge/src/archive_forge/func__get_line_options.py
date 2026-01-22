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
def _get_line_options(self):
    if self._line_options is None:
        self._line_options = kw = {'font_size': self.font_size, 'font_name': self.font_name, 'font_context': self.font_context, 'font_family': self.font_family, 'text_language': self.text_language, 'base_direction': self.base_direction, 'anchor_x': 'left', 'anchor_y': 'top', 'padding_x': 0, 'padding_y': 0, 'padding': (0, 0)}
        self._label_cached = Label(**kw)
    return self._line_options