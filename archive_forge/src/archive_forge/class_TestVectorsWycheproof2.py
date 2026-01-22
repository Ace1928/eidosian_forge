import json
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
class TestVectorsWycheproof2(unittest.TestCase):

    def __init__(self):
        unittest.TestCase.__init__(self)
        self._id = 'None'

    def setUp(self):
        self.tv = load_test_vectors_wycheproof(('Cipher', 'wycheproof'), 'aead_aes_siv_cmac_test.json', 'Wycheproof AEAD SIV')

    def shortDescription(self):
        return self._id

    def test_encrypt(self, tv):
        self._id = 'Wycheproof Encrypt AEAD-AES-SIV Test #' + str(tv.id)
        cipher = AES.new(tv.key, AES.MODE_SIV, nonce=tv.iv)
        cipher.update(tv.aad)
        ct, tag = cipher.encrypt_and_digest(tv.msg)
        if tv.valid:
            self.assertEqual(ct, tv.ct)
            self.assertEqual(tag, tv.tag)

    def test_decrypt(self, tv):
        self._id = 'Wycheproof Decrypt AEAD-AES-SIV Test #' + str(tv.id)
        cipher = AES.new(tv.key, AES.MODE_SIV, nonce=tv.iv)
        cipher.update(tv.aad)
        try:
            pt = cipher.decrypt_and_verify(tv.ct, tv.tag)
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid
            self.assertEqual(pt, tv.msg)

    def runTest(self):
        for tv in self.tv:
            self.test_encrypt(tv)
            self.test_decrypt(tv)