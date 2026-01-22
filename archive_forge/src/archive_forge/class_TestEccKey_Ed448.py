import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Hash import SHAKE128
class TestEccKey_Ed448(unittest.TestCase):

    def test_private_key(self):
        seed = unhexlify('4adf5d37ac6785e83e99a924f92676d366a78690af59c92b6bdf14f9cdbcf26fdad478109607583d633b60078d61d51d81b7509c5433b0d4c9')
        Px = 325446217306090953437856287370448806488502865468978757199471133175786932082059105257667529930261589134008879049952195686473111433291287
        Py = 448740337158713015934986081827345765524730480111986550601798045359153740662802087188188588374402113723130746791122877945925962315214529
        key = EccKey(curve='Ed448', seed=seed)
        self.assertEqual(key.seed, seed)
        self.assertEqual(key.d, 501087328496366969140677053865823318065705212104179624634307461836797857615836433313593441736527960295998751889288610349510870818185088)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ.x, Px)
        self.assertEqual(key.pointQ.y, Py)
        point = EccPoint(Px, Py, 'ed448')
        key = EccKey(curve='Ed448', seed=seed, point=point)
        self.assertEqual(key.d, 501087328496366969140677053865823318065705212104179624634307461836797857615836433313593441736527960295998751889288610349510870818185088)
        self.assertTrue(key.has_private())
        self.assertEqual(key.pointQ, point)
        key = EccKey(curve='ed448', seed=seed)
        self.assertRaises(ValueError, EccKey, curve='ed448', d=1)

    def test_public_key(self):
        point = EccPoint(_curves['ed448'].Gx, _curves['ed448'].Gy, curve='ed448')
        key = EccKey(curve='ed448', point=point)
        self.assertFalse(key.has_private())
        self.assertEqual(key.pointQ, point)

    def test_public_key_derived(self):
        priv_key = EccKey(curve='ed448', seed=b'H' * 57)
        pub_key = priv_key.public_key()
        self.assertFalse(pub_key.has_private())
        self.assertEqual(priv_key.pointQ, pub_key.pointQ)

    def test_invalid_seed(self):
        self.assertRaises(ValueError, lambda: EccKey(curve='ed448', seed=b'H' * 56))

    def test_equality(self):
        private_key = ECC.construct(seed=b'H' * 57, curve='Ed448')
        private_key2 = ECC.construct(seed=b'H' * 57, curve='ed448')
        private_key3 = ECC.construct(seed=b'C' * 57, curve='Ed448')
        public_key = private_key.public_key()
        public_key2 = private_key2.public_key()
        public_key3 = private_key3.public_key()
        self.assertEqual(private_key, private_key2)
        self.assertNotEqual(private_key, private_key3)
        self.assertEqual(public_key, public_key2)
        self.assertNotEqual(public_key, public_key3)
        self.assertNotEqual(public_key, private_key)

    def test_name_consistency(self):
        key = ECC.generate(curve='ed448')
        self.assertIn("curve='Ed448'", repr(key))
        self.assertEqual(key.curve, 'Ed448')
        self.assertEqual(key.public_key().curve, 'Ed448')