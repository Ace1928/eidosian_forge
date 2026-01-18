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
@_register_writer(1)
def write_byte(self, data):
    if isinstance(data, IFDRational):
        data = int(data)
    if isinstance(data, int):
        data = bytes((data,))
    return data