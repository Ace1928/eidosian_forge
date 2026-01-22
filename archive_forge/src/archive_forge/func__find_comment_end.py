from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
def _find_comment_end(self, block, start=0):
    a = block.find(b'\n', start)
    b = block.find(b'\r', start)
    return min(a, b) if a * b > 0 else max(a, b)