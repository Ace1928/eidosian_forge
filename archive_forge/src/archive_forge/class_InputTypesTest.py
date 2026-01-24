import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
class InputTypesTest(unittest.TestCase):
    """Check the various input types that CRC functions can accept."""
    msg = b'CatMouse989284321'
    check_crc_names = ['crc-aug-ccitt', 'x-25', 'crc-32']
    array_check_types = ['B', 'H', 'I', 'L']

    def test_bytearray_input(self):
        """Test that bytearray inputs are accepted, as an example
        of a type that implements the buffer protocol."""
        for crc_name in self.check_crc_names:
            crcfun = mkPredefinedCrcFun(crc_name)
            for i in range(len(self.msg) + 1):
                test_msg = self.msg[:i]
                bytes_answer = crcfun(test_msg)
                bytearray_answer = crcfun(bytearray(test_msg))
                self.assertEqual(bytes_answer, bytearray_answer)

    def test_array_input(self):
        """Test that array inputs are accepted, as an example
        of a type that implements the buffer protocol."""
        for crc_name in self.check_crc_names:
            crcfun = mkPredefinedCrcFun(crc_name)
            for i in range(len(self.msg) + 1):
                test_msg = self.msg[:i]
                bytes_answer = crcfun(test_msg)
                for array_type in self.array_check_types:
                    if i % array(array_type).itemsize == 0:
                        test_array = array(array_type, test_msg)
                        array_answer = crcfun(test_array)
                        self.assertEqual(bytes_answer, array_answer)

    def test_unicode_input(self):
        """Test that Unicode input raises TypeError"""
        for crc_name in self.check_crc_names:
            crcfun = mkPredefinedCrcFun(crc_name)
            with self.assertRaises(TypeError):
                crcfun('123456789')