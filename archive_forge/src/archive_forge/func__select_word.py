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
def _select_word(self, delimiters=u' .,:;!?\'"<>()[]{}'):
    cindex = self.cursor_index()
    col = self.cursor_col
    line = self._lines[self.cursor_row]
    start = max(0, len(line[:col]) - max((line[:col].rfind(s) for s in delimiters)) - 1)
    end = min((line[col:].find(s) if line[col:].find(s) > -1 else len(line) - col for s in delimiters))
    Clock.schedule_once(lambda dt: self.select_text(cindex - start, cindex + end))