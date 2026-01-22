import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccModule_P521(unittest.TestCase):

    def test_generate(self):
        curve = _curves['p521']
        key = ECC.generate(curve='P-521')
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, EccPoint(curve.Gx, curve.Gy, 'p521') * key.d)
        ECC.generate(curve='secp521r1')
        ECC.generate(curve='prime521v1')

    def test_construct(self):
        curve = _curves['p521']
        key = ECC.construct(curve='P-521', d=1)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, _curves['p521'].G)
        key = ECC.construct(curve='P-521', point_x=curve.Gx, point_y=curve.Gy)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, curve.G)
        ECC.construct(curve='p521', d=1)
        ECC.construct(curve='secp521r1', d=1)
        ECC.construct(curve='prime521v1', d=1)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['p521'].Gx, point_y=_curves['p521'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='P-521', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='P-521', d=2, **coordG)