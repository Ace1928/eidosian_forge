from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
class Reply(Response):

    def __init__(self, unpacker):
        Response.__init__(self, unpacker)
        resp = unpacker.cast('xcb_generic_reply_t *')
        self.length = resp.length