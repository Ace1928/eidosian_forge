import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def translate_matrix(m: Matrix, v: Point) -> Matrix:
    """Translates a matrix by (x, y)."""
    a, b, c, d, e, f = m
    x, y = v
    return (a, b, c, d, x * a + y * c + e, x * b + y * d + f)