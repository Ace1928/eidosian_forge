import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
def test_pai(self):
    pai = EccPoint(0, 1, curve='Ed448')
    self.assertTrue(pai.is_point_at_infinity())
    self.assertEqual(pai, pai.point_at_infinity())