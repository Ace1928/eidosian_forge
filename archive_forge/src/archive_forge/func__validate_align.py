from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_align(self, val):
    try:
        assert val in ['l', 'c', 'r']
    except AssertionError:
        msg = f'Alignment {val} is invalid, use l, c or r'
        raise ValueError(msg)