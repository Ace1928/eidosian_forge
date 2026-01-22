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
def _draw_selection(self, pos, size, line_num, selection_start, selection_end, lines, get_text_width, tab_width, label_cached, width_minus_padding, padding_left, padding_right, x, canvas_add, selection_color):
    selection_start_col, selection_start_row = selection_start
    selection_end_col, selection_end_row = selection_end
    if not selection_start_row <= line_num <= selection_end_row:
        return
    x, y = pos
    w, h = size
    beg = x
    end = x + w
    if line_num == selection_start_row:
        line = lines[line_num]
        beg -= self.scroll_x
        beg += get_text_width(line[:selection_start_col], tab_width, label_cached)
    if line_num == selection_end_row:
        line = lines[line_num]
        end = x - self.scroll_x + get_text_width(line[:selection_end_col], tab_width, label_cached)
    beg = boundary(beg, x, x + width_minus_padding)
    end = boundary(end, x, x + width_minus_padding)
    if beg == end:
        return
    canvas_add(Color(*selection_color, group='selection'))
    canvas_add(Rectangle(pos=(beg, y), size=(end - beg, h), group='selection'))