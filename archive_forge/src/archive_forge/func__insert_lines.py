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
def _insert_lines(self, start, finish, len_lines, _lines_flags, _lines, _lines_labels, _line_rects):
    self_lines_flags = self._lines_flags
    _lins_flags = []
    _lins_flags.extend(self_lines_flags[:start])
    if len_lines:
        if start:
            _lines_flags[0] = self_lines_flags[start]
        _lins_flags.extend(_lines_flags)
    _lins_flags.extend(self_lines_flags[finish:])
    self._lines_flags = _lins_flags
    _lins_lbls = []
    _lins_lbls.extend(self._lines_labels[:start])
    if len_lines:
        _lins_lbls.extend(_lines_labels)
    _lins_lbls.extend(self._lines_labels[finish:])
    self._lines_labels = _lins_lbls
    _lins_rcts = []
    _lins_rcts.extend(self._lines_rects[:start])
    if len_lines:
        _lins_rcts.extend(_line_rects)
    _lins_rcts.extend(self._lines_rects[finish:])
    self._lines_rects = _lins_rcts
    _lins = []
    _lins.extend(self._lines[:start])
    if len_lines:
        _lins.extend(_lines)
    _lins.extend(self._lines[finish:])
    self._lines[:] = _lins