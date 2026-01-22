from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _str_block_width(val):
    import wcwidth
    return wcwidth.wcswidth(_re.sub('', val))