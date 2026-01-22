import unittest
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, SHA224, SHA256, SHA384, SHA512
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pss
from Cryptodome.Signature import PKCS1_PSS
from Cryptodome.Signature.pss import MGF1
class TestVectorsPSSWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = 'None'

    def add_tests(self, filename):

        def filter_rsa(group):
            return RSA.import_key(group['keyPem'])

        def filter_sha(group):
            return get_hash_module(group['sha'])

        def filter_type(group):
            type_name = group['type']
            if type_name not in ('RsassaPssVerify',):
                raise ValueError('Unknown type name ' + type_name)

        def filter_slen(group):
            return group['sLen']

        def filter_mgf(group):
            mgf = group['mgf']
            if mgf not in ('MGF1',):
                raise ValueError('Unknown MGF ' + mgf)
            mgf1_hash = get_hash_module(group['mgfSha'])

            def mgf(x, y, mh=mgf1_hash):
                return MGF1(x, y, mh)
            return mgf
        result = load_test_vectors_wycheproof(('Signature', 'wycheproof'), filename, 'Wycheproof PSS signature (%s)' % filename, group_tag={'key': filter_rsa, 'hash_module': filter_sha, 'sLen': filter_slen, 'mgf': filter_mgf, 'type': filter_type})
        return result

    def setUp(self):
        self.tv = []
        self.add_tests('rsa_pss_2048_sha1_mgf1_20_test.json')
        self.add_tests('rsa_pss_2048_sha256_mgf1_0_test.json')
        self.add_tests('rsa_pss_2048_sha256_mgf1_32_test.json')
        self.add_tests('rsa_pss_2048_sha512_256_mgf1_28_test.json')
        self.add_tests('rsa_pss_2048_sha512_256_mgf1_32_test.json')
        self.add_tests('rsa_pss_3072_sha256_mgf1_32_test.json')
        self.add_tests('rsa_pss_4096_sha256_mgf1_32_test.json')
        self.add_tests('rsa_pss_4096_sha512_mgf1_32_test.json')
        self.add_tests('rsa_pss_misc_test.json')

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn('Wycheproof warning: %s (%s)' % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = 'Wycheproof RSA PSS Test #%d (%s)' % (tv.id, tv.comment)
        hashed_msg = tv.hash_module.new(tv.msg)
        signer = pss.new(tv.key, mask_func=tv.mgf, salt_bytes=tv.sLen)
        try:
            signature = signer.verify(hashed_msg, tv.sig)
        except ValueError as e:
            if tv.warning:
                return
            assert not tv.valid
        else:
            assert tv.valid
            self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)