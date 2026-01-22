from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class VoidCookie(Cookie):

    def reply(self):
        raise XcffibException('No reply for this message type')