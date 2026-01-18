from __future__ import print_function
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome import Random
from Cryptodome.Cipher import PKCS1_v1_5 as PKCS
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def test_return_type(self):
    pt = b'XYZ'
    cipher = PKCS.new(self.key1024)
    ct = cipher.encrypt(pt)
    self.assertTrue(isinstance(ct, bytes))
    pt2 = cipher.decrypt(ct, b'\xaa' * 3)
    self.assertTrue(isinstance(pt2, bytes))