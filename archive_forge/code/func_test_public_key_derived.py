import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def test_public_key_derived(self):
    priv_key = EccKey(curve='P-521', d=3)
    pub_key = priv_key.public_key()
    self.assertFalse(pub_key.has_private())
    self.assertEqual(priv_key.pointQ, pub_key.pointQ)