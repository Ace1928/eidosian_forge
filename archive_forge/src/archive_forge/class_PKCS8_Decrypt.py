import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.IO import PKCS8
from Cryptodome.Util.asn1 import DerNull
class PKCS8_Decrypt(unittest.TestCase):

    def setUp(self):
        self.oid_key = oid_key
        self.clear_key = txt2bin(clear_key)
        self.wrapped_clear_key = txt2bin(wrapped_clear_key)
        self.wrapped_enc_keys = []
        for t in wrapped_enc_keys:
            self.wrapped_enc_keys.append((t[0], t[1], txt2bin(t[2]), txt2bin(t[3]), txt2bin(t[4])))

    def test1(self):
        """Verify unwrapping w/o encryption"""
        res1, res2, res3 = PKCS8.unwrap(self.wrapped_clear_key)
        self.assertEqual(res1, self.oid_key)
        self.assertEqual(res2, self.clear_key)

    def test2(self):
        """Verify wrapping w/o encryption"""
        wrapped = PKCS8.wrap(self.clear_key, self.oid_key)
        res1, res2, res3 = PKCS8.unwrap(wrapped)
        self.assertEqual(res1, self.oid_key)
        self.assertEqual(res2, self.clear_key)

    def test3(self):
        """Verify unwrapping with encryption"""
        for t in self.wrapped_enc_keys:
            res1, res2, res3 = PKCS8.unwrap(t[4], b'TestTest')
            self.assertEqual(res1, self.oid_key)
            self.assertEqual(res2, self.clear_key)

    def test4(self):
        """Verify wrapping with encryption"""
        for t in self.wrapped_enc_keys:
            if t[0] == 'skip encryption':
                continue
            rng = Rng(t[2] + t[3])
            params = {'iteration_count': t[1]}
            wrapped = PKCS8.wrap(self.clear_key, self.oid_key, b('TestTest'), protection=t[0], prot_params=params, key_params=DerNull(), randfunc=rng)
            self.assertEqual(wrapped, t[4])

    def test_import_botan_keys(self):
        botan_scrypt_der = txt2bin(botan_scrypt)
        key1 = PKCS8.unwrap(botan_scrypt_der, b'your_password')
        botan_pbkdf2_der = txt2bin(botan_pbkdf2)
        key2 = PKCS8.unwrap(botan_pbkdf2_der, b'your_password')
        self.assertEqual(key1, key2)