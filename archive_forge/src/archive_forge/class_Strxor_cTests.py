import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
class Strxor_cTests(unittest.TestCase):

    def test1(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        result = unhexlify(b'be72dbc2a48c0d9e1708')
        self.assertEqual(strxor_c(term1, 65), result)

    def test2(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        self.assertEqual(strxor_c(term1, 0), term1)

    def test3(self):
        self.assertEqual(strxor_c(b'', 90), b'')

    def test_wrong_range(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        self.assertRaises(ValueError, strxor_c, term1, -1)
        self.assertRaises(ValueError, strxor_c, term1, 256)

    def test_bytearray(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term1_ba = bytearray(term1)
        result = unhexlify(b'be72dbc2a48c0d9e1708')
        self.assertEqual(strxor_c(term1_ba, 65), result)

    def test_memoryview(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        term1_mv = memoryview(term1)
        result = unhexlify(b'be72dbc2a48c0d9e1708')
        self.assertEqual(strxor_c(term1_mv, 65), result)

    def test_output_bytearray(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        original_term1 = term1[:]
        expected_result = unhexlify(b'be72dbc2a48c0d9e1708')
        output = bytearray(len(term1))
        result = strxor_c(term1, 65, output=output)
        self.assertEqual(result, None)
        self.assertEqual(output, expected_result)
        self.assertEqual(term1, original_term1)

    def test_output_memoryview(self):
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        original_term1 = term1[:]
        expected_result = unhexlify(b'be72dbc2a48c0d9e1708')
        output = memoryview(bytearray(len(term1)))
        result = strxor_c(term1, 65, output=output)
        self.assertEqual(result, None)
        self.assertEqual(output, expected_result)
        self.assertEqual(term1, original_term1)

    def test_output_overlapping_bytearray(self):
        """Verify result can be stored in overlapping memory"""
        term1 = bytearray(unhexlify(b'ff339a83e5cd4cdf5649'))
        expected_xor = unhexlify(b'be72dbc2a48c0d9e1708')
        result = strxor_c(term1, 65, output=term1)
        self.assertEqual(result, None)
        self.assertEqual(term1, expected_xor)

    def test_output_overlapping_memoryview(self):
        """Verify result can be stored in overlapping memory"""
        term1 = memoryview(bytearray(unhexlify(b'ff339a83e5cd4cdf5649')))
        expected_xor = unhexlify(b'be72dbc2a48c0d9e1708')
        result = strxor_c(term1, 65, output=term1)
        self.assertEqual(result, None)
        self.assertEqual(term1, expected_xor)

    def test_output_ro_bytes(self):
        """Verify result cannot be stored in read-only memory"""
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        self.assertRaises(TypeError, strxor_c, term1, 65, output=term1)

    def test_output_ro_memoryview(self):
        """Verify result cannot be stored in read-only memory"""
        term1 = memoryview(unhexlify(b'ff339a83e5cd4cdf5649'))
        term2 = unhexlify(b'383d4ba020573314395b')
        self.assertRaises(TypeError, strxor_c, term1, 65, output=term1)

    def test_output_incorrect_length(self):
        """Verify result cannot be stored in memory of incorrect length"""
        term1 = unhexlify(b'ff339a83e5cd4cdf5649')
        output = bytearray(len(term1) - 1)
        self.assertRaises(ValueError, strxor_c, term1, 65, output=output)