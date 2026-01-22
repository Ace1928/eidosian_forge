from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class CffiUnpacker(Unpacker):

    def __init__(self, cdata, known_max=None):
        self.cdata = cdata
        Unpacker.__init__(self, known_max)

    def _resize(self, increment):
        if self.offset + increment > self.size:
            if self.known_max is not None:
                assert self.size + increment <= self.known_max
            self.size = self.offset + increment
            self.buf = ffi.buffer(self.cdata, self.size)

    def copy(self):
        new = CffiUnpacker(self.cdata, self.known_max)
        new.offset = self.offset
        new.size = self.size
        return new