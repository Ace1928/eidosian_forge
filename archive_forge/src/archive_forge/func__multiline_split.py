from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _multiline_split(self, text):
    split_character = '\n' if isinstance(text, str) else b'\n'
    return text.split(split_character)