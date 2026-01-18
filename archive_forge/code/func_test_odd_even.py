import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_odd_even(self):
    v1, v2, v3, v4, v5 = self.Integers(0, 4, 17, -4, -17)
    self.assertTrue(v1.is_even())
    self.assertTrue(v2.is_even())
    self.assertFalse(v3.is_even())
    self.assertTrue(v4.is_even())
    self.assertFalse(v5.is_even())
    self.assertFalse(v1.is_odd())
    self.assertFalse(v2.is_odd())
    self.assertTrue(v3.is_odd())
    self.assertFalse(v4.is_odd())
    self.assertTrue(v5.is_odd())