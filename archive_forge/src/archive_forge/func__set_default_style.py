from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_default_style(self) -> None:
    self.header = True
    self.border = True
    self._hrules = FRAME
    self._vrules = ALL
    self.padding_width = 1
    self.left_padding_width = 1
    self.right_padding_width = 1
    self.vertical_char = '|'
    self.horizontal_char = '-'
    self._horizontal_align_char = None
    self.junction_char = '+'
    self._top_junction_char = None
    self._bottom_junction_char = None
    self._right_junction_char = None
    self._left_junction_char = None
    self._top_right_junction_char = None
    self._top_left_junction_char = None
    self._bottom_right_junction_char = None
    self._bottom_left_junction_char = None