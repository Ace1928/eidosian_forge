import sys
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math._IntegerNative import IntegerNative
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