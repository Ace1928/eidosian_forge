import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_random_range(self):
    func = IntegerNative.random_range
    for x in range(200):
        a = func(min_inclusive=1, max_inclusive=15)
        self.assertTrue(1 <= a <= 15)
    for x in range(200):
        a = func(min_inclusive=1, max_exclusive=15)
        self.assertTrue(1 <= a < 15)
    self.assertRaises(ValueError, func, min_inclusive=1, max_inclusive=2, max_exclusive=3)
    self.assertRaises(ValueError, func, max_inclusive=2, max_exclusive=3)