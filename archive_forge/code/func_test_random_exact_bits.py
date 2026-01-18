import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
def test_random_exact_bits(self):
    for _ in range(1000):
        a = IntegerNative.random(exact_bits=8)
        self.assertFalse(a < 128)
        self.assertFalse(a >= 256)
    for bits_value in range(1024, 1024 + 8):
        a = IntegerNative.random(exact_bits=bits_value)
        self.assertFalse(a < 2 ** (bits_value - 1))
        self.assertFalse(a >= 2 ** bits_value)