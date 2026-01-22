from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
class C_Test(unittest.TestCase):

    def test_ints(self):
        for i in range(512):
            for name in 'ABCDEFGHI':
                b = BITS()
                setattr(b, name, i)
                self.assertEqual(getattr(b, name), func(byref(b), name.encode('ascii')))

    @support.skip_if_sanitizer(ub=True)
    def test_shorts(self):
        b = BITS()
        name = 'M'
        if func(byref(b), name.encode('ascii')) == 999:
            self.skipTest('Compiler does not support signed short bitfields')
        for i in range(256):
            for name in 'MNOPQRS':
                b = BITS()
                setattr(b, name, i)
                self.assertEqual(getattr(b, name), func(byref(b), name.encode('ascii')))