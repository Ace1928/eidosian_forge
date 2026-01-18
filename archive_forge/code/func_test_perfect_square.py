import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_perfect_square(self):
    self.assertFalse(self.Integer(-9).is_perfect_square())
    self.assertTrue(self.Integer(0).is_perfect_square())
    self.assertTrue(self.Integer(1).is_perfect_square())
    self.assertFalse(self.Integer(2).is_perfect_square())
    self.assertFalse(self.Integer(3).is_perfect_square())
    self.assertTrue(self.Integer(4).is_perfect_square())
    self.assertTrue(self.Integer(39 * 39).is_perfect_square())
    self.assertFalse(self.Integer(39 * 39 + 1).is_perfect_square())
    for x in range(100, 1000):
        self.assertFalse(self.Integer(x ** 2 + 1).is_perfect_square())
        self.assertTrue(self.Integer(x ** 2).is_perfect_square())