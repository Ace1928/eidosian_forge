from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def wait_for_reply(self, sequence):
    error_p = ffi.new('xcb_generic_error_t **')
    data = lib.xcb_wait_for_reply(self._conn, sequence, error_p)
    data = ffi.gc(data, lib.free)
    try:
        self._process_error(error_p[0])
    finally:
        if error_p[0] != ffi.NULL:
            lib.free(error_p[0])
    if data == ffi.NULL:
        raise XcffibException('Bad sequence number %d' % sequence)
    reply = ffi.cast('xcb_generic_reply_t *', data)
    return CffiUnpacker(data, known_max=32 + reply.length * 4)