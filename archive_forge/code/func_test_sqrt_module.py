import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_sqrt_module(self):
    self.assertRaises(ValueError, self.Integer(5).sqrt, 0)
    self.assertRaises(ValueError, self.Integer(5).sqrt, -1)
    assert self.Integer(0).sqrt(5) == 0
    assert self.Integer(1).sqrt(5) in (1, 4)
    for p in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
        for i in range(0, p):
            square = i ** 2 % p
            res = self.Integer(square).sqrt(p)
            assert res in (i, p - i)
    self.assertRaises(ValueError, self.Integer(2).sqrt, 11)
    self.assertRaises(ValueError, self.Integer(4).sqrt, 10)
    assert self.Integer(5 - 11).sqrt(11) in (4, 7)
    assert self.Integer(5 + 11).sqrt(11) in (4, 7)