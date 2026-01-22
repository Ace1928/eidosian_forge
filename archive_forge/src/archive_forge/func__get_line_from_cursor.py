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
def _get_line_from_cursor(self, start, new_text, lines=None, lines_flags=None):
    if lines is None:
        lines = self._lines
    if lines_flags is None:
        lines_flags = self._lines_flags
    finish = start
    _next = start + 1
    if start > 0 and lines_flags[start] != FL_IS_LINEBREAK:
        start -= 1
        new_text = lines[start] + new_text
    i = _next
    for i in range(_next, len(lines_flags)):
        if lines_flags[i] == FL_IS_LINEBREAK:
            finish = i - 1
            break
    else:
        finish = i
    new_text = new_text + u''.join(lines[_next:finish + 1])
    lines, lines_flags = self._split_smart(new_text)
    len_lines = max(1, len(lines))
    return (start, finish, lines, lines_flags, len_lines)