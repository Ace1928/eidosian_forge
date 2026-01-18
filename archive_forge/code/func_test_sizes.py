from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
def test_sizes(self):
    dll = CDLL(_ctypes_test.__file__)
    for i in range(1, 11):
        fields = [(f'f{f}', c_char) for f in range(1, i + 1)]

        class S(Structure):
            _fields_ = fields
        f = getattr(dll, f'TestSize{i}')
        f.restype = S
        res = f()
        for i, f in enumerate(fields):
            value = getattr(res, f[0])
            expected = bytes([ord('a') + i])
            self.assertEqual(value, expected)