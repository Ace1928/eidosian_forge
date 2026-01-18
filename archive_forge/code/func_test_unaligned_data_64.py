import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_unaligned_data_64(self):
    cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
    for wrong_length in range(1, 8):
        self.assertRaises(ValueError, cipher.encrypt, b'5' * wrong_length)
    cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
    for wrong_length in range(1, 8):
        self.assertRaises(ValueError, cipher.decrypt, b'5' * wrong_length)