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
def _handle_shortcut(self, key):
    if key == ord('a'):
        self.select_all()
    elif key == ord('c'):
        self.copy()
    if not self._editable:
        return
    if key == ord('x'):
        self._cut(self.selection_text)
    elif key == ord('v'):
        self.paste()
    elif key == ord('z'):
        self.do_undo()
    elif key == ord('r'):
        self.do_redo()