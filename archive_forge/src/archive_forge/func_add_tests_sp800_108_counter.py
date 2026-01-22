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
def add_tests_sp800_108_counter(cls):
    test_vectors_sp800_108_counter = load_test_vectors(('Protocol',), 'KDF_SP800_108_COUNTER.txt', 'NIST SP 800 108 KDF Counter Mode', {'count': lambda x: int(x)}) or []
    mac_type = None
    for idx, tv in enumerate(test_vectors_sp800_108_counter):
        if isinstance(tv, str):
            res = re.match('\\[HMAC-(SHA-[0-9]+)\\]', tv)
            if res:
                hash_name = res.group(1).replace('-', '')
                hash_module = load_hash_by_name(hash_name)
                mac_type = 'hmac'
                continue
            res = re.match('\\[CMAC-AES-128\\]', tv)
            if res:
                mac_type = 'cmac'
                continue
            assert res
        if mac_type == 'hmac':

            def prf(s, x, hash_module=hash_module):
                return HMAC.new(s, x, hash_module).digest()
        elif mac_type == 'cmac':

            def prf(s, x, hash_module=hash_module):
                return CMAC.new(s, x, AES).digest()
            continue

        def kdf_test(self, prf=prf, kin=tv.kin, label=tv.label, context=tv.context, kout=tv.kout, count=tv.count):
            result = SP800_108_Counter(kin, len(kout), prf, 1, label, context)
            assert len(result) == len(kout)
            self.assertEqual(result, kout)
        setattr(cls, 'test_kdf_sp800_108_counter_%d' % idx, kdf_test)