import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
def test_output_param(self):
    pt = b'5' * 128
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    ct = cipher.encrypt(pt)
    tag = cipher.digest()
    output = bytearray(128)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    res = cipher.encrypt(pt, output=output)
    self.assertEqual(ct, output)
    self.assertEqual(res, None)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    res = cipher.decrypt(ct, output=output)
    self.assertEqual(pt, output)
    self.assertEqual(res, None)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    res, tag_out = cipher.encrypt_and_digest(pt, output=output)
    self.assertEqual(ct, output)
    self.assertEqual(res, None)
    self.assertEqual(tag, tag_out)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    res = cipher.decrypt_and_verify(ct, tag, output=output)
    self.assertEqual(pt, output)
    self.assertEqual(res, None)