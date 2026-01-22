from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class Unpacker(object):

    def __init__(self, known_max=None):
        self.size = 0
        self.offset = 0
        self.known_max = known_max
        if self.known_max is not None:
            self._resize(known_max)

    def pad(self, thing):
        if isinstance(thing, type) and issubclass(thing, (Struct, Union)):
            if hasattr(thing, 'fixed_size'):
                size = thing.fixed_size
            else:
                size = 4
        else:
            size = struct.calcsize(thing)
        self.offset += type_pad(size, self.offset)

    def unpack(self, fmt, increment=True):
        fmt = '=' + fmt
        size = struct.calcsize(fmt)
        if size > self.size - self.offset:
            self._resize(size)
        ret = struct.unpack_from(fmt, self.buf, self.offset)
        if increment:
            self.offset += size
        return ret

    def cast(self, typ):
        assert self.offset == 0
        return ffi.cast(typ, self.cdata)

    def copy(self):
        raise NotImplementedError

    @classmethod
    def synthetic(cls, data, format):
        self = cls.__new__(cls)
        self.__init__(len(data))
        self.buf = data
        self.offset = 0
        self.size = len(data)
        return self