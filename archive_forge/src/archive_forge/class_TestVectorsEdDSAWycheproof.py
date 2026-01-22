import unittest
from binascii import unhexlify
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import eddsa
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long
class TestVectorsEdDSAWycheproof(unittest.TestCase):

    def add_tests(self, filename):

        def pk(group):
            elem = group['key']['pk']
            return unhexlify(elem)

        def sk(group):
            elem = group['key']['sk']
            return unhexlify(elem)
        result = load_test_vectors_wycheproof(('Signature', 'wycheproof'), filename, 'Wycheproof ECDSA signature (%s)' % filename, group_tag={'pk': pk, 'sk': sk})
        self.tv += result

    def setUp(self):
        self.tv = []
        self.add_tests('eddsa_test.json')
        self.add_tests('ed448_test.json')

    def test_sign(self, tv):
        if not tv.valid:
            return
        self._id = 'Wycheproof EdDSA Sign Test #%d (%s, %s)' % (tv.id, tv.comment, tv.filename)
        key = eddsa.import_private_key(tv.sk)
        signer = eddsa.new(key, 'rfc8032')
        signature = signer.sign(tv.msg)
        self.assertEqual(signature, tv.sig)

    def test_verify(self, tv):
        self._id = 'Wycheproof EdDSA Verify Test #%d (%s, %s)' % (tv.id, tv.comment, tv.filename)
        key = eddsa.import_public_key(tv.pk)
        verifier = eddsa.new(key, 'rfc8032')
        try:
            verifier.verify(tv.msg, tv.sig)
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid

    def runTest(self):
        for tv in self.tv:
            self.test_sign(tv)
            self.test_verify(tv)