import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_nonce_attribute(self):
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    self.assertEqual(cipher.nonce, self.nonce_96)
    nonce1 = AES.new(self.key_128, AES.MODE_OCB).nonce
    nonce2 = AES.new(self.key_128, AES.MODE_OCB).nonce
    self.assertEqual(len(nonce1), 15)
    self.assertNotEqual(nonce1, nonce2)