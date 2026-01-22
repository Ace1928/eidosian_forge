import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def apply_matrix_norm(m: Matrix, v: Point) -> Point:
    """Equivalent to apply_matrix_pt(M, (p,q)) - apply_matrix_pt(M, (0,0))"""
    a, b, c, d, e, f = m
    p, q = v
    return (a * p + c * q, b * p + d * q)