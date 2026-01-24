import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
class TestEccKey_Ed25519(unittest.TestCase):

    def test_private_key(self):
        seed = unhexlify('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60')
        Px = 38815646466658113194383306759739515082307681141926459231621296960732224964046
        Py = 11903303657706407974989296177215005343713679411332034699907763981919547054807
        key = EccKey(curve='Ed25519', seed=seed)
        self.assertEqual(key.seed, seed)
        self.assertEqual(key.d, 36144925721603089288594284515452164870581325872720374094707712194495455132720)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ.x, Px)
        self.assertEqual(key.pointQ.y, Py)
        point = EccPoint(Px, Py, 'ed25519')
        key = EccKey(curve='Ed25519', seed=seed, point=point)
        self.assertEqual(key.d, 36144925721603089288594284515452164870581325872720374094707712194495455132720)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, point)
        key = EccKey(curve='ed25519', seed=seed)
        self.assertRaises(ValueError, EccKey, curve='ed25519', d=1)

    def test_public_key(self):
        point = EccPoint(_curves['ed25519'].Gx, _curves['ed25519'].Gy, curve='ed25519')
        key = EccKey(curve='ed25519', point=point)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, point)

    def test_public_key_derived(self):
        priv_key = EccKey(curve='ed25519', seed=b'H' * 32)
        pub_key = priv_key.public_key()
        self.assertFalse(pub_key.has_private())
        self.assertEqual(priv_key.pointQ, pub_key.pointQ)

    def test_invalid_seed(self):
        self.assertRaises(ValueError, lambda: EccKey(curve='ed25519', seed=b'H' * 31))

    def test_equality(self):
        private_key = ECC.construct(seed=b'H' * 32, curve='Ed25519')
        private_key2 = ECC.construct(seed=b'H' * 32, curve='ed25519')
        private_key3 = ECC.construct(seed=b'C' * 32, curve='Ed25519')
        public_key = private_key.public_key()
        public_key2 = private_key2.public_key()
        public_key3 = private_key3.public_key()
        self.assertEqual(private_key, private_key2)
        self.assertNotEqual(private_key, private_key3)
        self.assertEqual(public_key, public_key2)
        self.assertNotEqual(public_key, public_key3)
        self.assertNotEqual(public_key, private_key)

    def test_name_consistency(self):
        key = ECC.generate(curve='ed25519')
        self.assertIn("curve='Ed25519'", repr(key))
        self.assertEqual(key.curve, 'Ed25519')
        self.assertEqual(key.public_key().curve, 'Ed25519')