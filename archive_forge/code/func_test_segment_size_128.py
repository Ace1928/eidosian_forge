import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_segment_size_128(self):
    for bits in range(8, 129, 8):
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=bits)
    for bits in (0, 7, 9, 127, 129):
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CFB, self.iv_128, segment_size=bits)