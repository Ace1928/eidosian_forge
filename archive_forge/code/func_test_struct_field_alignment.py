import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_struct_field_alignment(self):
    if sys.byteorder == 'little':
        base = BigEndianStructure
        fmt = '>bxhid'
    else:
        base = LittleEndianStructure
        fmt = '<bxhid'

    class S(base):
        _fields_ = [('b', c_byte), ('h', c_short), ('i', c_int), ('d', c_double)]
    s1 = S(18, 4660, 305419896, 3.14)
    s2 = struct.pack(fmt, 18, 4660, 305419896, 3.14)
    self.assertEqual(bin(s1), bin(s2))