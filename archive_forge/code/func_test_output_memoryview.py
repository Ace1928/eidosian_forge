import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
def test_output_memoryview(self):
    term1 = unhexlify(b'ff339a83e5cd4cdf5649')
    original_term1 = term1[:]
    expected_result = unhexlify(b'be72dbc2a48c0d9e1708')
    output = memoryview(bytearray(len(term1)))
    result = strxor_c(term1, 65, output=output)
    self.assertEqual(result, None)
    self.assertEqual(output, expected_result)
    self.assertEqual(term1, original_term1)