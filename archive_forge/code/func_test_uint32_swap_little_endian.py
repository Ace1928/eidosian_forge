from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
@need_symbol('c_uint32')
def test_uint32_swap_little_endian(self):

    class Little(LittleEndianStructure):
        _fields_ = [('a', c_uint32, 24), ('b', c_uint32, 4), ('c', c_uint32, 4)]
    b = bytearray(4)
    x = Little.from_buffer(b)
    x.a = 11259375
    x.b = 1
    x.c = 2
    self.assertEqual(b, b'\xef\xcd\xab!')