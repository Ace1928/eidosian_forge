from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _set_markdown_style(self) -> None:
    self.header = True
    self.border = True
    self._hrules = None
    self.padding_width = 1
    self.left_padding_width = 1
    self.right_padding_width = 1
    self.vertical_char = '|'
    self.junction_char = '|'
    self._horizontal_align_char = ':'