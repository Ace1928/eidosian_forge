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
def test_simple_structs(self):
    for code, tp in self.formats.items():

        class X(Structure):
            _fields_ = [('x', c_char), ('y', tp)]
        self.assertEqual((sizeof(X), code), (calcsize('c%c0%c' % (code, code)), code))