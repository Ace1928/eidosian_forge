import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
def test_output_incorrect_length(self):
    """Verify result cannot be stored in memory of incorrect length"""
    term1 = unhexlify(b'ff339a83e5cd4cdf5649')
    output = bytearray(len(term1) - 1)
    self.assertRaises(ValueError, strxor_c, term1, 65, output=output)