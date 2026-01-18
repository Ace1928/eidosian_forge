from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def to_atoms(self):
    """ A helper for converting a List of chars to an array of atoms """
    return struct.unpack('<' + '%dI' % (len(self) // 4), b''.join(self))