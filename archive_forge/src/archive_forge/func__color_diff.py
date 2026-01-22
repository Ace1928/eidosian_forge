from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _color_diff(color1, color2):
    """
    Uses 1-norm distance to calculate difference between two values.
    """
    if isinstance(color2, tuple):
        return sum((abs(color1[i] - color2[i]) for i in range(0, len(color2))))
    else:
        return abs(color1 - color2)