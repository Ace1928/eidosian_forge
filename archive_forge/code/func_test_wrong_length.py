import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
def test_wrong_length(self):
    term1 = unhexlify(b'ff339a83e5cd4cdf5649')
    term2 = unhexlify(b'ff339a83e5cd4cdf564990')
    self.assertRaises(ValueError, strxor, term1, term2)