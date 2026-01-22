import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
class TestVectorsECDSAWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, slow_tests):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._slow_tests = slow_tests
        self._id = 'None'

    def add_tests(self, filename):

        def filter_ecc(group):
            if group['key']['curve'] in ('secp224k1', 'secp256k1', 'brainpoolP224r1', 'brainpoolP224t1', 'brainpoolP256r1', 'brainpoolP256t1', 'brainpoolP320r1', 'brainpoolP320t1', 'brainpoolP384r1', 'brainpoolP384t1', 'brainpoolP512r1', 'brainpoolP512t1'):
                return None
            return ECC.import_key(group['keyPem'])

        def filter_sha(group):
            return get_hash_module(group['sha'])

        def filter_encoding(group):
            encoding_name = group['type']
            if encoding_name == 'EcdsaVerify':
                return 'der'
            elif encoding_name == 'EcdsaP1363Verify':
                return 'binary'
            else:
                raise ValueError('Unknown signature type ' + encoding_name)
        result = load_test_vectors_wycheproof(('Signature', 'wycheproof'), filename, 'Wycheproof ECDSA signature (%s)' % filename, group_tag={'key': filter_ecc, 'hash_module': filter_sha, 'encoding': filter_encoding})
        self.tv += result

    def setUp(self):
        self.tv = []
        self.add_tests('ecdsa_secp224r1_sha224_p1363_test.json')
        self.add_tests('ecdsa_secp224r1_sha224_test.json')
        if self._slow_tests:
            self.add_tests('ecdsa_secp224r1_sha256_p1363_test.json')
            self.add_tests('ecdsa_secp224r1_sha256_test.json')
            self.add_tests('ecdsa_secp224r1_sha3_224_test.json')
            self.add_tests('ecdsa_secp224r1_sha3_256_test.json')
            self.add_tests('ecdsa_secp224r1_sha3_512_test.json')
            self.add_tests('ecdsa_secp224r1_sha512_p1363_test.json')
            self.add_tests('ecdsa_secp224r1_sha512_test.json')
            self.add_tests('ecdsa_secp256r1_sha256_p1363_test.json')
            self.add_tests('ecdsa_secp256r1_sha256_test.json')
            self.add_tests('ecdsa_secp256r1_sha3_256_test.json')
            self.add_tests('ecdsa_secp256r1_sha3_512_test.json')
            self.add_tests('ecdsa_secp256r1_sha512_p1363_test.json')
        self.add_tests('ecdsa_secp256r1_sha512_test.json')
        if self._slow_tests:
            self.add_tests('ecdsa_secp384r1_sha3_384_test.json')
            self.add_tests('ecdsa_secp384r1_sha3_512_test.json')
            self.add_tests('ecdsa_secp384r1_sha384_p1363_test.json')
            self.add_tests('ecdsa_secp384r1_sha384_test.json')
            self.add_tests('ecdsa_secp384r1_sha512_p1363_test.json')
        self.add_tests('ecdsa_secp384r1_sha512_test.json')
        if self._slow_tests:
            self.add_tests('ecdsa_secp521r1_sha3_512_test.json')
            self.add_tests('ecdsa_secp521r1_sha512_p1363_test.json')
        self.add_tests('ecdsa_secp521r1_sha512_test.json')
        self.add_tests('ecdsa_test.json')
        self.add_tests('ecdsa_webcrypto_test.json')

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn('Wycheproof warning: %s (%s)' % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = 'Wycheproof ECDSA Test #%d (%s, %s)' % (tv.id, tv.comment, tv.filename)
        if tv.key is None:
            return
        hashed_msg = tv.hash_module.new(tv.msg)
        signer = DSS.new(tv.key, 'fips-186-3', encoding=tv.encoding)
        try:
            signature = signer.verify(hashed_msg, tv.sig)
        except ValueError as e:
            if tv.warning:
                return
            if tv.comment == 'k*G has a large x-coordinate':
                return
            assert not tv.valid
        else:
            assert tv.valid
            self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)