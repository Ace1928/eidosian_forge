from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def type_pad(t, i):
    return -i & (3 if t > 4 else t - 1)