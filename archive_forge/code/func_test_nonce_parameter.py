import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
def test_nonce_parameter(self):
    cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64)
    self.assertEqual(cipher1.nonce, self.nonce_64)
    counter = Counter.new(64, prefix=self.nonce_64, initial_value=0)
    cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
    self.assertEqual(cipher1.nonce, cipher2.nonce)
    pt = get_tag_random('plaintext', 65536)
    self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))
    nonce1 = AES.new(self.key_128, AES.MODE_CTR).nonce
    nonce2 = AES.new(self.key_128, AES.MODE_CTR).nonce
    self.assertNotEqual(nonce1, nonce2)
    self.assertEqual(len(nonce1), 8)
    cipher = AES.new(self.key_128, AES.MODE_CTR, nonce=b'')
    self.assertEqual(b'', cipher.nonce)
    cipher.encrypt(b'0' * 300)
    self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, counter=self.ctr_128, nonce=self.nonce_64)