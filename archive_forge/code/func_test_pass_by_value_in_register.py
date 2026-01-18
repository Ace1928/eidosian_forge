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
def test_pass_by_value_in_register(self):

    class X(Structure):
        _fields_ = [('first', c_uint), ('second', c_uint)]
    s = X()
    s.first = 3735928559
    s.second = 3405691582
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_reg_struct_update_value
    func.argtypes = (X,)
    func.restype = None
    func(s)
    self.assertEqual(s.first, 3735928559)
    self.assertEqual(s.second, 3405691582)
    got = X.in_dll(dll, 'last_tfrsuv_arg')
    self.assertEqual(s.first, got.first)
    self.assertEqual(s.second, got.second)