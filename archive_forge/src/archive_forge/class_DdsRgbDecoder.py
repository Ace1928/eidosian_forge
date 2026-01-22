from __future__ import annotations
import io
import struct
import sys
from enum import IntEnum, IntFlag
from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32
class DdsRgbDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        bitcount, masks = self.args
        mask_offsets = []
        mask_totals = []
        for mask in masks:
            offset = 0
            if mask != 0:
                while mask >> offset + 1 << offset + 1 == mask:
                    offset += 1
            mask_offsets.append(offset)
            mask_totals.append(mask >> offset)
        data = bytearray()
        bytecount = bitcount // 8
        while len(data) < self.state.xsize * self.state.ysize * len(masks):
            value = int.from_bytes(self.fd.read(bytecount), 'little')
            for i, mask in enumerate(masks):
                masked_value = value & mask
                data += o8(int((masked_value >> mask_offsets[i]) / mask_totals[i] * 255))
        self.set_as_raw(bytes(data))
        return (-1, 0)