import array
import contextlib
import enum
import struct
def _Unpack(fmt, buf):
    return struct.unpack('<%s' % fmt[len(buf)], buf)[0]