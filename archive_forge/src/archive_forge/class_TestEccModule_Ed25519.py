import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
class TestEccModule_Ed25519(unittest.TestCase):

    def test_generate(self):
        key = ECC.generate(curve='Ed25519')
        self.assertTrue(key.has_private())
        point = EccPoint(_curves['Ed25519'].Gx, _curves['Ed25519'].Gy, curve='Ed25519') * key.d
        self.assertEqual(key.pointQ, point)
        key2 = ECC.generate(curve='Ed25519')
        self.assertNotEqual(key, key2)
        ECC.generate(curve='Ed25519')
        key1 = ECC.generate(curve='Ed25519', randfunc=SHAKE128.new().read)
        key2 = ECC.generate(curve='Ed25519', randfunc=SHAKE128.new().read)
        self.assertEqual(key1, key2)

    def test_construct(self):
        seed = unhexlify('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60')
        Px = 38815646466658113194383306759739515082307681141926459231621296960732224964046
        Py = 11903303657706407974989296177215005343713679411332034699907763981919547054807
        d = 36144925721603089288594284515452164870581325872720374094707712194495455132720
        point = EccPoint(Px, Py, curve='Ed25519')
        key = ECC.construct(curve='Ed25519', seed=seed)
        self.assertEqual(key.pointQ, point)
        self.assertTrue(key.has_private())
        key = ECC.construct(curve='Ed25519', point_x=Px, point_y=Py)
        self.assertEqual(key.pointQ, point)
        self.assertFalse(key.has_private())
        key = ECC.construct(curve='Ed25519', seed=seed, point_x=Px, point_y=Py)
        self.assertEqual(key.pointQ, point)
        self.assertTrue(key.has_private())
        key = ECC.construct(curve='ed25519', seed=seed)

    def test_negative_construct(self):
        coord = dict(point_x=10, point_y=4)
        coordG = dict(point_x=_curves['ed25519'].Gx, point_y=_curves['ed25519'].Gy)
        self.assertRaises(ValueError, ECC.construct, curve='Ed25519', **coord)
        self.assertRaises(ValueError, ECC.construct, curve='Ed25519', d=2, **coordG)
        self.assertRaises(ValueError, ECC.construct, curve='Ed25519', seed=b'H' * 31)