from ctypes import *
import unittest
import struct
def test_signed_values(self):
    for t, (l, h) in zip(signed_types, signed_ranges):
        self.assertEqual(t(l).value, l)
        self.assertEqual(t(h).value, h)