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
class scrypt_Tests(unittest.TestCase):
    data = (('', '', 16, 1, 1, '\n                    77 d6 57 62 38 65 7b 20 3b 19 ca 42 c1 8a 04 97\n                    f1 6b 48 44 e3 07 4a e8 df df fa 3f ed e2 14 42\n                    fc d0 06 9d ed 09 48 f8 32 6a 75 3a 0f c8 1f 17\n                    e8 d3 e0 fb 2e 0d 36 28 cf 35 e2 0c 38 d1 89 06\n                    '), ('password', 'NaCl', 1024, 8, 16, '\n                    fd ba be 1c 9d 34 72 00 78 56 e7 19 0d 01 e9 fe\n                    7c 6a d7 cb c8 23 78 30 e7 73 76 63 4b 37 31 62\n                    2e af 30 d9 2e 22 a3 88 6f f1 09 27 9d 98 30 da\n                    c7 27 af b9 4a 83 ee 6d 83 60 cb df a2 cc 06 40\n                    '), ('pleaseletmein', 'SodiumChloride', 16384, 8, 1, '\n                    70 23 bd cb 3a fd 73 48 46 1c 06 cd 81 fd 38 eb\n                    fd a8 fb ba 90 4f 8e 3e a9 b5 43 f6 54 5d a1 f2\n                    d5 43 29 55 61 3f 0f cf 62 d4 97 05 24 2a 9a f9\n                    e6 1e 85 dc 0d 65 1e 40 df cf 01 7b 45 57 58 87\n                    '), ('pleaseletmein', 'SodiumChloride', 1048576, 8, 1, '\n                    21 01 cb 9b 6a 51 1a ae ad db be 09 cf 70 f8 81\n                    ec 56 8d 57 4a 2f fd 4d ab e5 ee 98 20 ad aa 47\n                    8e 56 fd 8f 4b a5 d0 9f fa 1c 6d 92 7c 40 f4 c3\n                    37 30 40 49 e8 a9 52 fb cb f4 5c 6f a7 7a 41 a4\n                    '))

    def setUp(self):
        new_test_vectors = []
        for tv in self.data:
            new_tv = TestVector()
            new_tv.P = b(tv[0])
            new_tv.S = b(tv[1])
            new_tv.N = tv[2]
            new_tv.r = tv[3]
            new_tv.p = tv[4]
            new_tv.output = t2b(tv[5])
            new_tv.dkLen = len(new_tv.output)
            new_test_vectors.append(new_tv)
        self.data = new_test_vectors

    def test2(self):
        for tv in self.data:
            try:
                output = scrypt(tv.P, tv.S, tv.dkLen, tv.N, tv.r, tv.p)
            except ValueError as e:
                if ' 2 ' in str(e) and tv.N >= 1048576:
                    import warnings
                    warnings.warn('Not enough memory to unit test scrypt() with N=1048576', RuntimeWarning)
                    continue
                else:
                    raise e
            self.assertEqual(output, tv.output)

    def test3(self):
        ref = scrypt(b('password'), b('salt'), 12, 16, 1, 1)
        key1, key2 = scrypt(b('password'), b('salt'), 6, 16, 1, 1, 2)
        self.assertEqual((ref[:6], ref[6:]), (key1, key2))
        key1, key2, key3 = scrypt(b('password'), b('salt'), 4, 16, 1, 1, 3)
        self.assertEqual((ref[:4], ref[4:8], ref[8:]), (key1, key2, key3))