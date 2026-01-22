from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def _res_to_dpi(num, denom, exp):
    """Convert JPEG2000's (numerator, denominator, exponent-base-10) resolution,
    calculated as (num / denom) * 10^exp and stored in dots per meter,
    to floating-point dots per inch."""
    if denom != 0:
        return 254 * num * 10 ** exp / (10000 * denom)