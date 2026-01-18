from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@top_left_junction_char.setter
def top_left_junction_char(self, val) -> None:
    val = str(val)
    self._validate_option('top_left_junction_char', val)
    self._top_left_junction_char = val