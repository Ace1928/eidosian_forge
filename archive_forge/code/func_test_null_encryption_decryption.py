import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_null_encryption_decryption(self):
    for func in ('encrypt', 'decrypt'):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        result = getattr(cipher, func)(b(''))
        self.assertEqual(result, b(''))