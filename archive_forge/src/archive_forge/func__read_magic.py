from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
def _read_magic(self):
    magic = b''
    for _ in range(6):
        c = self.fp.read(1)
        if not c or c in b_whitespace:
            break
        magic += c
    return magic