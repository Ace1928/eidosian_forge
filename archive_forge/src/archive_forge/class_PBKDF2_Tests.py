import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
class PBKDF2_Tests(unittest.TestCase):
    _testData = (('password', '78578E5A5D63CB06', 24, 2048, SHA1, 'BFDE6BE94DF7E11DD409BCE20A0255EC327CB936FFE93643'), ('password', '73616c74', 20, 1, SHA1, '0c60c80f961f0e71f3a9b524af6012062fe037a6'), ('password', '73616c74', 20, 2, SHA1, 'ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957'), ('password', '73616c74', 20, 4096, SHA1, '4b007901b765489abead49d926f721d065a429c1'), ('passwordPASSWORDpassword', '73616c7453414c5473616c7453414c5473616c7453414c5473616c7453414c5473616c74', 25, 4096, SHA1, '3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038'), ('pass\x00word', '7361006c74', 16, 4096, SHA1, '56fa6aa75548099dcc37d7f03425e0c3'), ('passwd', '73616c74', 64, 1, SHA256, '55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783'), ('Password', '4e61436c', 64, 80000, SHA256, '4ddcd8f60b98be21830cee5ef22701f9641a4418d04c0414aeff08876b34ab56a1d425a1225833549adb841b51c9b3176a272bdebba1d078478f62b397f33c8d'))

    def test1(self):

        def prf_SHA1(p, s):
            return HMAC.new(p, s, SHA1).digest()

        def prf_SHA256(p, s):
            return HMAC.new(p, s, SHA256).digest()
        for i in range(len(self._testData)):
            v = self._testData[i]
            password = v[0]
            salt = t2b(v[1])
            out_len = v[2]
            iters = v[3]
            hash_mod = v[4]
            expected = t2b(v[5])
            if hash_mod is SHA1:
                res = PBKDF2(password, salt, out_len, iters)
                self.assertEqual(res, expected)
                res = PBKDF2(password, salt, out_len, iters, prf_SHA1)
                self.assertEqual(res, expected)
            else:
                res = PBKDF2(password, salt, out_len, iters, prf_SHA256)
                self.assertEqual(res, expected)

    def test2(self):

        def prf_SHA1(p, s):
            return HMAC.new(p, s, SHA1).digest()
        self.assertRaises(ValueError, PBKDF2, b('xxx'), b('yyy'), 16, 100, prf=prf_SHA1, hmac_hash_module=SHA1)

    def test3(self):
        password = b('xxx')
        salt = b('yyy')
        for hashmod in (MD5, SHA1, SHA224, SHA256, SHA384, SHA512):
            pr1 = PBKDF2(password, salt, 16, 100, prf=lambda p, s: HMAC.new(p, s, hashmod).digest())
            pr2 = PBKDF2(password, salt, 16, 100, hmac_hash_module=hashmod)
            self.assertEqual(pr1, pr2)

    def test4(self):
        k1 = PBKDF2('xxx', b('yyy'), 16, 10)
        k2 = PBKDF2(b('xxx'), b('yyy'), 16, 10)
        self.assertEqual(k1, k2)
        k1 = PBKDF2(b('xxx'), 'yyy', 16, 10)
        k2 = PBKDF2(b('xxx'), b('yyy'), 16, 10)
        self.assertEqual(k1, k2)