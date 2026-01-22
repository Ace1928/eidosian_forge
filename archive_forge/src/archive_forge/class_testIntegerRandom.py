import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
class testIntegerRandom(unittest.TestCase):

    def test_random_exact_bits(self):
        for _ in range(1000):
            a = IntegerNative.random(exact_bits=8)
            self.assertFalse(a < 128)
            self.assertFalse(a >= 256)
        for bits_value in range(1024, 1024 + 8):
            a = IntegerNative.random(exact_bits=bits_value)
            self.assertFalse(a < 2 ** (bits_value - 1))
            self.assertFalse(a >= 2 ** bits_value)

    def test_random_max_bits(self):
        flag = False
        for _ in range(1000):
            a = IntegerNative.random(max_bits=8)
            flag = flag or a < 128
            self.assertFalse(a >= 256)
        self.assertTrue(flag)
        for bits_value in range(1024, 1024 + 8):
            a = IntegerNative.random(max_bits=bits_value)
            self.assertFalse(a >= 2 ** bits_value)

    def test_random_bits_custom_rng(self):

        class CustomRNG(object):

            def __init__(self):
                self.counter = 0

            def __call__(self, size):
                self.counter += size
                return bchr(0) * size
        custom_rng = CustomRNG()
        a = IntegerNative.random(exact_bits=32, randfunc=custom_rng)
        self.assertEqual(custom_rng.counter, 4)

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