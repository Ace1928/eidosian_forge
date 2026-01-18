import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_simple_exponentiation(self):
    v1, v2, v3 = self.Integers(4, 3, -2)
    self.assertTrue(isinstance(v1 ** v2, self.Integer))
    self.assertEqual(v1 ** v2, 64)
    self.assertEqual(pow(v1, v2), 64)
    self.assertEqual(v1 ** 3, 64)
    self.assertEqual(pow(v1, 3), 64)
    self.assertEqual(v3 ** 2, 4)
    self.assertEqual(v3 ** 3, -8)
    self.assertRaises(ValueError, pow, v1, -3)