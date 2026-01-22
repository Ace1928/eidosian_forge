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
def _on_textinput_focused(self, instance, value, *largs):
    win = EventLoop.window
    self.cancel_selection()
    self._hide_cut_copy_paste(win)
    if value:
        if not (self.readonly or self.disabled) or (_is_desktop and self._keyboard_mode == 'system'):
            self._trigger_cursor_reset()
            self._editable = True
        else:
            self._editable = False
    else:
        self._do_blink_cursor_ev.cancel()
        self._hide_handles(win)