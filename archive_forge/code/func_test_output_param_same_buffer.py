import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
def test_output_param_same_buffer(self):
    pt = b'5' * 128
    cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
    ct = cipher.encrypt(pt)
    pt_ba = bytearray(pt)
    cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
    res = cipher.encrypt(pt_ba, output=pt_ba)
    self.assertEqual(ct, pt_ba)
    self.assertEqual(res, None)
    ct_ba = bytearray(ct)
    cipher = AES.new(b'4' * 16, self.aes_mode, iv=self.iv_128)
    res = cipher.decrypt(ct_ba, output=ct_ba)
    self.assertEqual(pt, ct_ba)
    self.assertEqual(res, None)