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
def _move_cursor_down(self, col, row, control, alt):
    if self.multiline and control:
        maxy = self.minimum_height - self.height
        self.scroll_y = max(0, min(maxy, self.scroll_y + self.line_height))
    elif not self.readonly and self.multiline and alt:
        self._shift_lines(1)
        return
    else:
        row = min(row + 1, len(self._lines) - 1)
        col = min(len(self._lines[row]), col)
    return (col, row)