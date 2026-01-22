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
def _update_graphics(self, *largs):
    """
        Update all the graphics according to the current internal values.
        """
    self.canvas.clear()
    line_height = self.line_height
    dy = line_height + self.line_spacing
    scroll_x = self.scroll_x
    scroll_y = self.scroll_y
    if not self._lines or (not self._lines[0] and len(self._lines) == 1):
        rects = self._hint_text_rects
        labels = self._hint_text_labels
        lines = self._hint_text_lines
    else:
        rects = self._lines_rects
        labels = self._lines_labels
        lines = self._lines
    padding_left, padding_top, padding_right, padding_bottom = self.padding
    x = self.x + padding_left
    y = self.top - padding_top + scroll_y
    miny = self.y + padding_bottom
    maxy = self.top - padding_top
    halign = self.halign
    base_dir = self.base_direction
    auto_halign_r = halign == 'auto' and base_dir and ('rtl' in base_dir)
    fst_visible_ln = None
    viewport_pos = (scroll_x, 0)
    for line_num, value in enumerate(lines):
        if miny < y < maxy + dy:
            if fst_visible_ln is None:
                fst_visible_ln = line_num
            y = self._draw_line(value, line_num, labels[line_num], viewport_pos, line_height, miny, maxy, x, y, base_dir, halign, rects, auto_halign_r)
        elif y <= miny:
            line_num -= 1
            break
        y -= dy
    if fst_visible_ln is not None:
        self._visible_lines_range = (fst_visible_ln, line_num + 1)
    else:
        self._visible_lines_range = (0, 0)
    self._update_graphics_selection()