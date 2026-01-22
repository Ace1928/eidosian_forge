from unittest import main, TestCase, TestSuite
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Protocol.SecretSharing import Shamir, _Element, \
class GF2_Tests(TestCase):

    def test_mult_gf2(self):
        x = _mult_gf2(0, 0)
        self.assertEqual(x, 0)
        x = _mult_gf2(34, 1)
        self.assertEqual(x, 34)
        z = 3
        y = _mult_gf2(z, z)
        self.assertEqual(y, 5)
        y = _mult_gf2(y, z)
        self.assertEqual(y, 15)
        y = _mult_gf2(y, z)
        self.assertEqual(y, 17)
        comps = [1, 4, 128, 2 ** 34]
        sum_comps = 1 + 4 + 128 + 2 ** 34
        y = 908
        z = _mult_gf2(sum_comps, y)
        w = 0
        for x in comps:
            w ^= _mult_gf2(x, y)
        self.assertEqual(w, z)

    def test_div_gf2(self):
        from Cryptodome.Util.number import size as deg
        x, y = _div_gf2(567, 7)
        self.assertTrue(deg(y) < deg(7))
        w = _mult_gf2(x, 7) ^ y
        self.assertEqual(567, w)
        x, y = _div_gf2(7, 567)
        self.assertEqual(x, 0)
        self.assertEqual(y, 7)