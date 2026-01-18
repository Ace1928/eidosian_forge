from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def to_utf8(self):
    return b''.join(self).decode('utf-8')