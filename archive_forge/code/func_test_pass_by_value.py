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
def test_pass_by_value(self):

    class Test(Structure):
        _fields_ = [('first', c_ulong), ('second', c_ulong), ('third', c_ulong)]
    s = Test()
    s.first = 3735928559
    s.second = 3405691582
    s.third = 195894762
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_large_struct_update_value
    func.argtypes = (Test,)
    func.restype = None
    func(s)
    self.assertEqual(s.first, 3735928559)
    self.assertEqual(s.second, 3405691582)
    self.assertEqual(s.third, 195894762)