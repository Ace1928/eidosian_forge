import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
class DSATest(unittest.TestCase):
    y = _sws('19131871 d75b1612 a819f29d 78d1b0d7 346f7aa7 7bb62a85\n                9bfd6c56 75da9d21 2d3a36ef 1672ef66 0b8c7c25 5cc0ec74\n                858fba33 f44c0669 9630a76b 030ee333')
    g = _sws('626d0278 39ea0a13 413163a5 5b4cb500 299d5522 956cefcb\n                3bff10f3 99ce2c2e 71cb9de5 fa24babf 58e5b795 21925c9c\n                c42e9f6f 464b088c c572af53 e6d78802')
    p = _sws('8df2a494 492276aa 3d25759b b06869cb eac0d83a fb8d0cf7\n                cbb8324f 0d7882e5 d0762fc5 b7210eaf c2e9adac 32ab7aac\n                49693dfb f83724c2 ec0736ee 31c80291')
    q = _sws('c773218c 737ec8ee 993b4f2d ed30f48e dace915f')
    x = _sws('2070b322 3dba372f de1c0ffc 7b2e3b49 8b260614')
    k = _sws('358dad57 1462710f 50e254cf 1a376b2b deaadfbf')
    k_inverse = _sws('0d516729 8202e49b 4116ac10 4fc3f415 ae52f917')
    m = b2a_hex(b('abc'))
    m_hash = _sws('a9993e36 4706816a ba3e2571 7850c26c 9cd0d89d')
    r = _sws('8bac1ab6 6410435c b7181f95 b16ab97c 92b341c0')
    s = _sws('41e2345f 1f56df24 58f426d1 55b4ba2d b6dcd8c8')

    def setUp(self):
        global DSA, Random, bytes_to_long, size
        from Cryptodome.PublicKey import DSA
        from Cryptodome import Random
        from Cryptodome.Util.number import bytes_to_long, inverse, size
        self.dsa = DSA

    def test_generate_1arg(self):
        """DSA (default implementation) generated key (1 argument)"""
        dsaObj = self.dsa.generate(1024)
        self._check_private_key(dsaObj)
        pub = dsaObj.public_key()
        self._check_public_key(pub)

    def test_generate_2arg(self):
        """DSA (default implementation) generated key (2 arguments)"""
        dsaObj = self.dsa.generate(1024, Random.new().read)
        self._check_private_key(dsaObj)
        pub = dsaObj.public_key()
        self._check_public_key(pub)

    def test_construct_4tuple(self):
        """DSA (default implementation) constructed key (4-tuple)"""
        y, g, p, q = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q)]
        dsaObj = self.dsa.construct((y, g, p, q))
        self._test_verification(dsaObj)

    def test_construct_5tuple(self):
        """DSA (default implementation) constructed key (5-tuple)"""
        y, g, p, q, x = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q, self.x)]
        dsaObj = self.dsa.construct((y, g, p, q, x))
        self._test_signing(dsaObj)
        self._test_verification(dsaObj)

    def test_construct_bad_key4(self):
        y, g, p, q = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q)]
        tup = (y, g, p + 1, q)
        self.assertRaises(ValueError, self.dsa.construct, tup)
        tup = (y, g, p, q + 1)
        self.assertRaises(ValueError, self.dsa.construct, tup)
        tup = (y, 1, p, q)
        self.assertRaises(ValueError, self.dsa.construct, tup)

    def test_construct_bad_key5(self):
        y, g, p, q, x = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q, self.x)]
        tup = (y, g, p, q, x + 1)
        self.assertRaises(ValueError, self.dsa.construct, tup)
        tup = (y, g, p, q, q + 10)
        self.assertRaises(ValueError, self.dsa.construct, tup)

    def _check_private_key(self, dsaObj):
        self.assertEqual(1, dsaObj.has_private())
        self.assertEqual(1, dsaObj.can_sign())
        self.assertEqual(0, dsaObj.can_encrypt())
        self.assertEqual(1, dsaObj.p > dsaObj.q)
        self.assertEqual(160, size(dsaObj.q))
        self.assertEqual(0, (dsaObj.p - 1) % dsaObj.q)
        self.assertEqual(dsaObj.y, pow(dsaObj.g, dsaObj.x, dsaObj.p))
        self.assertEqual(1, 0 < dsaObj.x < dsaObj.q)

    def _check_public_key(self, dsaObj):
        k = bytes_to_long(a2b_hex(self.k))
        m_hash = bytes_to_long(a2b_hex(self.m_hash))
        self.assertEqual(0, dsaObj.has_private())
        self.assertEqual(1, dsaObj.can_sign())
        self.assertEqual(0, dsaObj.can_encrypt())
        self.assertEqual(0, hasattr(dsaObj, 'x'))
        self.assertEqual(1, dsaObj.p > dsaObj.q)
        self.assertEqual(160, size(dsaObj.q))
        self.assertEqual(0, (dsaObj.p - 1) % dsaObj.q)
        self.assertRaises(TypeError, dsaObj._sign, m_hash, k)
        self.assertEqual(dsaObj.public_key() == dsaObj.public_key(), True)
        self.assertEqual(dsaObj.public_key() != dsaObj.public_key(), False)
        self.assertEqual(dsaObj.public_key(), dsaObj.publickey())

    def _test_signing(self, dsaObj):
        k = bytes_to_long(a2b_hex(self.k))
        m_hash = bytes_to_long(a2b_hex(self.m_hash))
        r = bytes_to_long(a2b_hex(self.r))
        s = bytes_to_long(a2b_hex(self.s))
        r_out, s_out = dsaObj._sign(m_hash, k)
        self.assertEqual((r, s), (r_out, s_out))

    def _test_verification(self, dsaObj):
        m_hash = bytes_to_long(a2b_hex(self.m_hash))
        r = bytes_to_long(a2b_hex(self.r))
        s = bytes_to_long(a2b_hex(self.s))
        self.assertTrue(dsaObj._verify(m_hash, (r, s)))
        self.assertFalse(dsaObj._verify(m_hash + 1, (r, s)))

    def test_repr(self):
        y, g, p, q = [bytes_to_long(a2b_hex(param)) for param in (self.y, self.g, self.p, self.q)]
        dsaObj = self.dsa.construct((y, g, p, q))
        repr(dsaObj)