import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
def test_output_param_neg(self):
    LEN_PT = 16
    pt = b'5' * LEN_PT
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    ct = cipher.encrypt(pt)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * LEN_PT)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * LEN_PT)
    shorter_output = bytearray(LEN_PT - 1)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
    cipher = AES.new(self.key_128, AES.MODE_EAX, nonce=self.nonce_96)
    self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)