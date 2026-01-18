import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_size_in_bytes(self):
    v1, v2, v3, v4, v5, v6 = self.Integers(0, 1, 255, 511, 65536, -9)
    self.assertEqual(v1.size_in_bytes(), 1)
    self.assertEqual(v2.size_in_bytes(), 1)
    self.assertEqual(v3.size_in_bytes(), 1)
    self.assertEqual(v4.size_in_bytes(), 2)
    self.assertEqual(v5.size_in_bytes(), 3)
    self.assertRaises(ValueError, v6.size_in_bits)