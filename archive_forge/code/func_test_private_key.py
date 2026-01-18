import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def test_private_key(self):
    p521 = _curves['p521']
    key = EccKey(curve='P-521', d=1)
    self.assertEqual(key.d, 1)
    self.assertTrue(key.has_private())
    self.assertEqual(key.pointQ.x, p521.Gx)
    self.assertEqual(key.pointQ.y, p521.Gy)
    point = EccPoint(p521.Gx, p521.Gy, 'p521')
    key = EccKey(curve='P-521', d=1, point=point)
    self.assertEqual(key.d, 1)
    self.assertTrue(key.has_private())
    self.assertEqual(key.pointQ, point)
    key = EccKey(curve='p521', d=1)
    key = EccKey(curve='secp521r1', d=1)
    key = EccKey(curve='prime521v1', d=1)