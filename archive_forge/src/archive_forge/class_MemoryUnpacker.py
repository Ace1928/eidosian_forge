from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class MemoryUnpacker(Unpacker):

    def __init__(self, buf):
        self.buf = buf
        Unpacker.__init__(self, len(self.buf))

    def _resize(self, increment):
        if self.size + increment > self.known_max:
            raise XcffibException('resizing memory buffer to be too big')
        self.size += increment

    def copy(self):
        new = MemoryUnpacker(self.buf)
        new.offset = self.offset
        new.size = self.size
        return new