import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Cipher import DES3
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.Util.py3compat import bchr, tostr
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
def test_parity_option3(self):
    before_3k = unhexlify('AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCC')
    after_3k = DES3.adjust_key_parity(before_3k)
    self.assertEqual(after_3k, unhexlify('ABABABABABABABABBABABABABABABABACDCDCDCDCDCDCDCD'))