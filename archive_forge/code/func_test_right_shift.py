import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_right_shift(self):
    v1, v2, v3 = self.Integers(16, 1, -16)
    self.assertEqual(v1 >> 0, v1)
    self.assertTrue(isinstance(v1 >> v2, self.Integer))
    self.assertEqual(v1 >> v2, 8)
    self.assertEqual(v1 >> 1, 8)
    self.assertRaises(ValueError, lambda: v1 >> -1)
    self.assertEqual(v1 >> 2 ** 1000, 0)
    self.assertEqual(v3 >> 1, -8)
    self.assertEqual(v3 >> 2 ** 1000, -1)