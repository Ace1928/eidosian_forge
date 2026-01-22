from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class String8(ValueField):
    structcode = None

    def __init__(self, name, pad=1):
        ValueField.__init__(self, name)
        self.pad = pad

    def pack_value(self, val):
        slen = len(val)
        if _PY3 and type(val) is str:
            val = val.encode('UTF-8')
        if self.pad:
            return (val + b'\x00' * ((4 - slen % 4) % 4), slen, None)
        else:
            return (val, slen, None)

    def parse_binary_value(self, data, display, length, format):
        if length is None:
            try:
                return (data.decode('UTF-8'), b'')
            except UnicodeDecodeError:
                return (data, b'')
        if self.pad:
            slen = length + (4 - length % 4) % 4
        else:
            slen = length
        s = data[:length]
        try:
            s = s.decode('UTF-8')
        except UnicodeDecodeError:
            pass
        return (s, data[slen:])