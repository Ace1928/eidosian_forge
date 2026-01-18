import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_unsupported_restype_1(self):
    prototype = self.functype.__func__(POINTER(c_double))
    self.assertRaises(TypeError, prototype, lambda: None)