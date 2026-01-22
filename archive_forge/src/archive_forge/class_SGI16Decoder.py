from __future__ import annotations
import os
import struct
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
class SGI16Decoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        rawmode, stride, orientation = self.args
        pagesize = self.state.xsize * self.state.ysize
        zsize = len(self.mode)
        self.fd.seek(512)
        for band in range(zsize):
            channel = Image.new('L', (self.state.xsize, self.state.ysize))
            channel.frombytes(self.fd.read(2 * pagesize), 'raw', 'L;16B', stride, orientation)
            self.im.putband(channel.im, band)
        return (-1, 0)