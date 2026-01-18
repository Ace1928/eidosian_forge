from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def to_cffi(self):
    c_key = ffi.new('struct xcb_extension_t *')
    c_key.name = name = ffi.new('char[]', self.name.encode())
    cffi_explicit_lifetimes[c_key] = name
    c_key.global_id = 0
    return c_key