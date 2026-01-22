from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
def _read_block(self):
    return self.fd.read(ImageFile.SAFEBLOCK)