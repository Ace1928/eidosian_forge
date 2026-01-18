from ctypes import *
import unittest
import struct
def test_unsigned_values(self):
    for t, (l, h) in zip(unsigned_types, unsigned_ranges):
        self.assertEqual(t(l).value, l)
        self.assertEqual(t(h).value, h)