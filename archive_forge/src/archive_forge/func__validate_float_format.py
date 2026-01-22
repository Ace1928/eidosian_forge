from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_float_format(self, name, val):
    if val == '':
        return
    try:
        assert isinstance(val, str)
        assert '.' in val
        bits = val.split('.')
        assert len(bits) <= 2
        assert bits[0] == '' or bits[0].isdigit()
        assert bits[1] == '' or bits[1].isdigit() or (bits[1][-1] == 'f' and bits[1].rstrip('f').isdigit())
    except AssertionError:
        msg = f'Invalid value for {name}. Must be a float format string.'
        raise ValueError(msg)