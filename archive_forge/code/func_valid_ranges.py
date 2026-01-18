from ctypes import *
import unittest
import struct
def valid_ranges(*types):
    result = []
    for t in types:
        fmt = t._type_
        size = struct.calcsize(fmt)
        a = struct.unpack(fmt, (b'\x00' * 32)[:size])[0]
        b = struct.unpack(fmt, (b'\xff' * 32)[:size])[0]
        c = struct.unpack(fmt, (b'\x7f' + b'\x00' * 32)[:size])[0]
        d = struct.unpack(fmt, (b'\x80' + b'\xff' * 32)[:size])[0]
        result.append((min(a, b, c, d), max(a, b, c, d)))
    return result