from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
def _limit_signed_rational(val, max_val, min_val):
    frac = Fraction(val)
    n_d = (frac.numerator, frac.denominator)
    if min(n_d) < min_val:
        n_d = _limit_rational(val, abs(min_val))
    if max(n_d) > max_val:
        val = Fraction(*n_d)
        n_d = _limit_rational(val, max_val)
    return n_d