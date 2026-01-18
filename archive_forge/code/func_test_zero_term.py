import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (create_string_buffer,
from Cryptodome.Math._IntegerCustom import _raw_montgomery
def test_zero_term(self):
    numbers_len = (modulus1.bit_length() + 7) // 8
    expect = b'\x00' * numbers_len
    self.assertEqual(expect, monty_mult(256, 0, modulus1))
    self.assertEqual(expect, monty_mult(0, 256, modulus1))