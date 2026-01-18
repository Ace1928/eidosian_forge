import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
def test_output_ro_memoryview(self):
    """Verify result cannot be stored in read-only memory"""
    term1 = memoryview(unhexlify(b'ff339a83e5cd4cdf5649'))
    term2 = unhexlify(b'383d4ba020573314395b')
    self.assertRaises(TypeError, strxor_c, term1, 65, output=term1)