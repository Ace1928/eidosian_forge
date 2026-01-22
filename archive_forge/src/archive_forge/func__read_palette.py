from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def _read_palette(self):
    ret = []
    for i in range(256):
        try:
            b, g, r, a = struct.unpack('<4B', self._safe_read(4))
        except struct.error:
            break
        ret.append((b, g, r, a))
    return ret