import re
from kivy.properties import dpi2px
from kivy.parser import parse_color
from kivy.logger import Logger
from kivy.core.text import Label, LabelBase
from kivy.core.text.text_layout import layout_text, LayoutWord, LayoutLine
from copy import copy
from functools import partial
def _push_style(self, k):
    if k not in self._style_stack:
        self._style_stack[k] = []
    self._style_stack[k].append(self.options[k])