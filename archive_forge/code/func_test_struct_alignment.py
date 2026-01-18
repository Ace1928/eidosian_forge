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
def test_struct_alignment(self):

    class X(Structure):
        _fields_ = [('x', c_char * 3)]
    self.assertEqual(alignment(X), calcsize('s'))
    self.assertEqual(sizeof(X), calcsize('3s'))

    class Y(Structure):
        _fields_ = [('x', c_char * 3), ('y', c_int)]
    self.assertEqual(alignment(Y), alignment(c_int))
    self.assertEqual(sizeof(Y), calcsize('3si'))

    class SI(Structure):
        _fields_ = [('a', X), ('b', Y)]
    self.assertEqual(alignment(SI), max(alignment(Y), alignment(X)))
    self.assertEqual(sizeof(SI), calcsize('3s0i 3si 0i'))

    class IS(Structure):
        _fields_ = [('b', Y), ('a', X)]
    self.assertEqual(alignment(SI), max(alignment(X), alignment(Y)))
    self.assertEqual(sizeof(IS), calcsize('3si 3s 0i'))

    class XX(Structure):
        _fields_ = [('a', X), ('b', X)]
    self.assertEqual(alignment(XX), alignment(X))
    self.assertEqual(sizeof(XX), calcsize('3s 3s 0s'))