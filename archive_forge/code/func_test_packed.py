import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
def test_packed(self):

    class X(Structure):
        _fields_ = [('a', c_byte), ('b', c_longlong)]
        _pack_ = 1
    self.assertEqual(sizeof(X), 9)
    self.assertEqual(X.b.offset, 1)

    class X(Structure):
        _fields_ = [('a', c_byte), ('b', c_longlong)]
        _pack_ = 2
    self.assertEqual(sizeof(X), 10)
    self.assertEqual(X.b.offset, 2)
    import struct
    longlong_size = struct.calcsize('q')
    longlong_align = struct.calcsize('bq') - longlong_size

    class X(Structure):
        _fields_ = [('a', c_byte), ('b', c_longlong)]
        _pack_ = 4
    self.assertEqual(sizeof(X), min(4, longlong_align) + longlong_size)
    self.assertEqual(X.b.offset, min(4, longlong_align))

    class X(Structure):
        _fields_ = [('a', c_byte), ('b', c_longlong)]
        _pack_ = 8
    self.assertEqual(sizeof(X), min(8, longlong_align) + longlong_size)
    self.assertEqual(X.b.offset, min(8, longlong_align))
    d = {'_fields_': [('a', 'b'), ('b', 'q')], '_pack_': -1}
    self.assertRaises(ValueError, type(Structure), 'X', (Structure,), d)