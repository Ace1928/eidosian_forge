import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
class TestEccPoint(unittest.TestCase):

    def test_mix(self):
        p1 = ECC.generate(curve='P-256').pointQ
        p2 = ECC.generate(curve='P-384').pointQ
        try:
            p1 + p2
            assert False
        except ValueError as e:
            assert 'not on the same curve' in str(e)
        try:
            p1 += p2
            assert False
        except ValueError as e:
            assert 'not on the same curve' in str(e)

        class OtherKeyType:
            pass
        self.assertFalse(p1 == OtherKeyType())
        self.assertTrue(p1 != OtherKeyType())

    def test_repr(self):
        p1 = ECC.construct(curve='P-256', d=75467964919405407085864614198393977741148485328036093939970922195112333446269, point_x=20573031766139722500939782666697015100983491952082159880539639074939225934381, point_y=108863130203210779921520632367477406025152638284581252625277850513266505911389)
        self.assertEqual(repr(p1), "EccKey(curve='NIST P-256', point_x=20573031766139722500939782666697015100983491952082159880539639074939225934381, point_y=108863130203210779921520632367477406025152638284581252625277850513266505911389, d=75467964919405407085864614198393977741148485328036093939970922195112333446269)")