import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.py3compat import *
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Hash import SHAKE128
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math._IntegerCustom import _raw_montgomery
from Cryptodome.Random.random import StrongRandom
def test_zero_exp(self):
    base = 25711008708143844408671393477458601640355247900524685364822015
    result = monty_pow(base, 0, modulus1)
    self.assertEqual(result, 1)