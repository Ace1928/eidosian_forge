from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
def SOF(self, marker):
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    self._size = (i16(s, 3), i16(s, 1))
    self.bits = s[0]
    if self.bits != 8:
        msg = f'cannot handle {self.bits}-bit layers'
        raise SyntaxError(msg)
    self.layers = s[5]
    if self.layers == 1:
        self._mode = 'L'
    elif self.layers == 3:
        self._mode = 'RGB'
    elif self.layers == 4:
        self._mode = 'CMYK'
    else:
        msg = f'cannot handle {self.layers}-layer images'
        raise SyntaxError(msg)
    if marker in [65474, 65478, 65482, 65486]:
        self.info['progressive'] = self.info['progression'] = 1
    if self.icclist:
        self.icclist.sort()
        if self.icclist[0][13] == len(self.icclist):
            profile = [p[14:] for p in self.icclist]
            icc_profile = b''.join(profile)
        else:
            icc_profile = None
        self.info['icc_profile'] = icc_profile
        self.icclist = []
    for i in range(6, len(s), 3):
        t = s[i:i + 3]
        self.layer.append((t[0], t[1] // 16, t[1] & 15, t[2]))