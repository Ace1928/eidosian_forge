import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Cipher import DES3
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.Util.py3compat import bchr, tostr
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
def test_parity_option2(self):
    before_2k = unhexlify('CABF326FA56734324FFCCABCDEFACABF')
    after_2k = DES3.adjust_key_parity(before_2k)
    self.assertEqual(after_2k, unhexlify('CBBF326EA46734324FFDCBBCDFFBCBBF'))