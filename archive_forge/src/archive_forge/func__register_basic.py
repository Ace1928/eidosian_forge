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
def _register_basic(idx_fmt_name):
    from .TiffTags import TYPES
    idx, fmt, name = idx_fmt_name
    TYPES[idx] = name
    size = struct.calcsize('=' + fmt)
    _load_dispatch[idx] = (size, lambda self, data, legacy_api=True: self._unpack(f'{len(data) // size}{fmt}', data))
    _write_dispatch[idx] = lambda self, *values: b''.join((self._pack(fmt, value) for value in values))