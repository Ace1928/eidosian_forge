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
def _ensure_read(self, fp, size):
    ret = fp.read(size)
    if len(ret) != size:
        msg = f'Corrupt EXIF data.  Expecting to read {size} bytes but only got {len(ret)}. '
        raise OSError(msg)
    return ret