import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_size_in_bits(self):
    v1, v2, v3, v4 = self.Integers(0, 1, 256, -90)
    self.assertEqual(v1.size_in_bits(), 1)
    self.assertEqual(v2.size_in_bits(), 1)
    self.assertEqual(v3.size_in_bits(), 9)
    self.assertRaises(ValueError, v4.size_in_bits)