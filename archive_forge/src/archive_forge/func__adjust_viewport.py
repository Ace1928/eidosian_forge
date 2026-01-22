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
@triggered(timeout=-1)
def _adjust_viewport(self, cc, cr):
    padding_left = self.padding[0]
    padding_right = self.padding[2]
    viewport_width = self.width - padding_left - padding_right
    sx = self.scroll_x
    base_dir = self.base_direction or self._resolved_base_dir
    auto_halign_r = self.halign == 'auto' and base_dir and ('rtl' in base_dir)
    offset = self.cursor_offset()
    row_width = self._get_row_width(self.cursor_row)
    if offset - sx >= viewport_width:
        self.scroll_x = offset - viewport_width
    elif offset < sx + 1:
        self.scroll_x = offset
    viewport_scroll_x = row_width - viewport_width
    if not self.multiline and offset >= viewport_scroll_x and (self.scroll_x >= viewport_scroll_x) and (self.halign == 'center' or self.halign == 'right' or auto_halign_r):
        self.scroll_x = max(0, viewport_scroll_x)
    dy = self.line_height + self.line_spacing
    offsety = cr * dy
    padding_top = self.padding[1]
    padding_bottom = self.padding[3]
    viewport_height = self.height - padding_top - padding_bottom - dy
    sy = self.scroll_y
    if offsety > viewport_height + sy:
        self.scroll_y = offsety - viewport_height
    elif offsety < sy:
        self.scroll_y = offsety