import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def apply_matrix_pt(m: Matrix, v: Point) -> Point:
    a, b, c, d, e, f = m
    x, y = v
    'Applies a matrix to a point.'
    return (a * x + c * y + e, b * x + d * y + f)