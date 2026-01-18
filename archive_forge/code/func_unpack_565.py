from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def unpack_565(i):
    return ((i >> 11 & 31) << 3, (i >> 5 & 63) << 2, (i & 31) << 3)