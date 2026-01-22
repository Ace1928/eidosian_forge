from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class KeyboardMapping(ValueField):
    structcode = None

    def parse_binary_value(self, data, display, length, format):
        if length is None:
            dlen = len(data)
        else:
            dlen = 4 * length * format
        a = array(array_unsigned_codes[4], data[:dlen])
        ret = []
        for i in range(0, len(a), format):
            ret.append(a[i:i + format])
        return (ret, data[dlen:])

    def pack_value(self, value):
        keycodes = 0
        for v in value:
            keycodes = max(keycodes, len(v))
        a = array(array_unsigned_codes[4])
        for v in value:
            for k in v:
                a.append(k)
            for i in range(len(v), keycodes):
                a.append(X.NoSymbol)
        return (a.tobytes(), len(value), keycodes)