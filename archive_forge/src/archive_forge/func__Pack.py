import array
import contextlib
import enum
import struct
def _Pack(fmt, value, byte_width):
    return struct.pack('<%s' % fmt[byte_width], value)