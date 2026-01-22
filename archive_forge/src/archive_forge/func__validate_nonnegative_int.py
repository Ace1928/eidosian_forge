from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_nonnegative_int(self, name, val):
    try:
        assert int(val) >= 0
    except AssertionError:
        msg = f'Invalid value for {name}: {val}'
        raise ValueError(msg)