from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class BLPFormatError(NotImplementedError):
    pass