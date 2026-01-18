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
def test_positional_args(self):

    class W(Structure):
        _fields_ = [('a', c_int), ('b', c_int)]

    class X(W):
        _fields_ = [('c', c_int)]

    class Y(X):
        pass

    class Z(Y):
        _fields_ = [('d', c_int), ('e', c_int), ('f', c_int)]
    z = Z(1, 2, 3, 4, 5, 6)
    self.assertEqual((z.a, z.b, z.c, z.d, z.e, z.f), (1, 2, 3, 4, 5, 6))
    z = Z(1)
    self.assertEqual((z.a, z.b, z.c, z.d, z.e, z.f), (1, 0, 0, 0, 0, 0))
    self.assertRaises(TypeError, lambda: Z(1, 2, 3, 4, 5, 6, 7))