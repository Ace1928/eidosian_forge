import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
def test_variable_exponent(self):
    prng = create_rng(b('Test variable exponent'))
    for i in range(20):
        for j in range(7):
            modulus = prng.getrandbits(8 * 30) | 1
            base = prng.getrandbits(8 * 30) % modulus
            exponent = prng.getrandbits(i * 8 + j)
            expected = pow(base, exponent, modulus)
            result = monty_pow(base, exponent, modulus)
            self.assertEqual(result, expected)
            exponent ^= (1 << i * 8 + j) - 1
            expected = pow(base, exponent, modulus)
            result = monty_pow(base, exponent, modulus)
            self.assertEqual(result, expected)