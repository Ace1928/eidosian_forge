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
class S2V_Tests(unittest.TestCase):
    _testData = [(('101112131415161718191a1b1c1d1e1f2021222324252627', '112233445566778899aabbccddee'), 'fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0', '85632d07c6e8f37f950acd320a2ecc93', AES), (('00112233445566778899aabbccddeeffdeaddadadeaddadaffeeddcc' + 'bbaa99887766554433221100', '102030405060708090a0', '09f911029d74e35bd84156c5635688c0', '7468697320697320736f6d6520706c61' + '696e7465787420746f20656e63727970' + '74207573696e67205349562d414553'), '7f7e7d7c7b7a79787776757473727170', '7bdb6e3b432667eb06f4d14bff2fbd0f', AES)]

    def test1(self):
        """Verify correctness of test vector"""
        for tv in self._testData:
            s2v = _S2V.new(t2b(tv[1]), tv[3])
            for s in tv[0]:
                s2v.update(t2b(s))
            result = s2v.derive()
            self.assertEqual(result, t2b(tv[2]))

    def test2(self):
        """Verify that no more than 127(AES) and 63(TDES)
        components are accepted."""
        key = bchr(0) * 8 + bchr(255) * 8
        for module in (AES, DES3):
            s2v = _S2V.new(key, module)
            max_comps = module.block_size * 8 - 1
            for i in range(max_comps):
                s2v.update(b('XX'))
            self.assertRaises(TypeError, s2v.update, b('YY'))