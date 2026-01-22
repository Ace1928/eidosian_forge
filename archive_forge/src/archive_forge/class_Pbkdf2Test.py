from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
class Pbkdf2Test(TestCase):
    """test pbkdf2() support"""
    descriptionPrefix = 'passlib.crypto.digest.pbkdf2_hmac() <backends: %s>' % ', '.join(PBKDF2_BACKENDS)
    pbkdf2_test_vectors = [(hb('cdedb5281bb2f801565a1122b2563515'), b'password', b'ATHENA.MIT.EDUraeburn', 1, 16), (hb('01dbee7f4a9e243e988b62c73cda935d'), b'password', b'ATHENA.MIT.EDUraeburn', 2, 16), (hb('01dbee7f4a9e243e988b62c73cda935da05378b93244ec8f48a99e61ad799d86'), b'password', b'ATHENA.MIT.EDUraeburn', 2, 32), (hb('5c08eb61fdf71e4e4ec3cf6ba1f5512ba7e52ddbc5e5142f708a31e2e62b1e13'), b'password', b'ATHENA.MIT.EDUraeburn', 1200, 32), (hb('d1daa78615f287e6a1c8b120d7062a493f98d203e6be49a6adf4fa574b6e64ee'), b'password', b'\x124VxxV4\x12', 5, 32), (hb('139c30c0966bc32ba55fdbf212530ac9c5ec59f1a452f5cc9ad940fea0598ed1'), b'X' * 64, b'pass phrase equals block size', 1200, 32), (hb('9ccad6d468770cd51b10e6a68721be611a8b4d282601db3b36be9246915ec82a'), b'X' * 65, b'pass phrase exceeds block size', 1200, 32), (hb('0c60c80f961f0e71f3a9b524af6012062fe037a6'), b'password', b'salt', 1, 20), (hb('ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957'), b'password', b'salt', 2, 20), (hb('4b007901b765489abead49d926f721d065a429c1'), b'password', b'salt', 4096, 20), (hb('3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038'), b'passwordPASSWORDpassword', b'saltSALTsaltSALTsaltSALTsaltSALTsalt', 4096, 25), (hb('56fa6aa75548099dcc37d7f03425e0c3'), b'pass\x00word', b'sa\x00lt', 4096, 16), (hb('887CFF169EA8335235D8004242AA7D6187A41E3187DF0CE14E256D85ED97A97357AAA8FF0A3871AB9EEFF458392F462F495487387F685B7472FC6C29E293F0A0'), b'hello', hb('9290F727ED06C38BA4549EF7DE25CF5642659211B7FC076F2D28FEFD71784BB8D8F6FB244A8CC5C06240631B97008565A120764C0EE9C2CB0073994D79080136'), 10000, 64, 'sha512'), (hb('55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783'), b'passwd', b'salt', 1, 64, 'sha256'), (hb('4ddcd8f60b98be21830cee5ef22701f9641a4418d04c0414aeff08876b34ab56a1d425a1225833549adb841b51c9b3176a272bdebba1d078478f62b397f33c8d'), b'Password', b'NaCl', 80000, 64, 'sha256'), (hb('120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b'), b'password', b'salt', 1, 32, 'sha256'), (hb('ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43'), b'password', b'salt', 2, 32, 'sha256'), (hb('c5e478d59288c841aa530db6845c4c8d962893a001ce4e11a4963873aa98134a'), b'password', b'salt', 4096, 32, 'sha256'), (hb('348c89dbcbd32b2f32d814b8116e84cf2b17347ebc1800181c4e2a1fb8dd53e1c635518c7dac47e9'), b'passwordPASSWORDpassword', b'saltSALTsaltSALTsaltSALTsaltSALTsalt', 4096, 40, 'sha256'), (hb('9e83f279c040f2a11aa4a02b24c418f2d3cb39560c9627fa4f47e3bcc2897c3d'), b'', b'salt', 1024, 32, 'sha256'), (hb('ea5808411eb0c7e830deab55096cee582761e22a9bc034e3ece925225b07bf46'), b'password', b'', 1024, 32, 'sha256'), (hb('89b69d0516f829893c696226650a8687'), b'pass\x00word', b'sa\x00lt', 4096, 16, 'sha256'), (hb('867f70cf1ade02cff3752599a3a53dc4af34c7a669815ae5d513554e1c8cf252'), b'password', b'salt', 1, 32, 'sha512'), (hb('e1d9c16aa681708a45f5c7c4e215ceb66e011a2e9f0040713f18aefdb866d53c'), b'password', b'salt', 2, 32, 'sha512'), (hb('d197b1b33db0143e018b12f3d1d1479e6cdebdcc97c5c0f87f6902e072f457b5'), b'password', b'salt', 4096, 32, 'sha512'), (hb('6e23f27638084b0f7ea1734e0d9841f55dd29ea60a834466f3396bac801fac1eeb63802f03a0b4acd7603e3699c8b74437be83ff01ad7f55dac1ef60f4d56480c35ee68fd52c6936'), b'passwordPASSWORDpassword', b'saltSALTsaltSALTsaltSALTsaltSALTsalt', 1, 72, 'sha512'), (hb('0c60c80f961f0e71f3a9b524af6012062fe037a6'), b'password', b'salt', 1, 20, 'sha1'), (hb('e248fb6b13365146f8ac6307cc222812'), b'secret', b'salt', 10, 16, 'sha1'), (hb('e248fb6b13365146f8ac6307cc2228127872da6d'), b'secret', b'salt', 10, None, 'sha1'), (hb('b1d5485772e6f76d5ebdc11b38d3eff0a5b2bd50dc11f937e86ecacd0cd40d1b9113e0734e3b76a3'), b'secret', b'salt', 62, 40, 'md5'), (hb('ea014cc01f78d3883cac364bb5d054e2be238fb0b6081795a9d84512126e3129062104d2183464c4'), b'secret', b'salt', 62, 40, 'md4')]

    def test_known(self):
        """test reference vectors"""
        for row in self.pbkdf2_test_vectors:
            correct, secret, salt, rounds, keylen = row[:5]
            digest = row[5] if len(row) == 6 else 'sha1'
            result = pbkdf2_hmac(digest, secret, salt, rounds, keylen)
            self.assertEqual(result, correct)

    def test_backends(self):
        """verify expected backends are present"""
        from passlib.crypto.digest import PBKDF2_BACKENDS
        try:
            import fastpbkdf2
            has_fastpbkdf2 = True
        except ImportError:
            has_fastpbkdf2 = False
        self.assertEqual('fastpbkdf2' in PBKDF2_BACKENDS, has_fastpbkdf2)
        try:
            from hashlib import pbkdf2_hmac
            has_hashlib_ssl = pbkdf2_hmac.__module__ != 'hashlib'
        except ImportError:
            has_hashlib_ssl = False
        self.assertEqual('hashlib-ssl' in PBKDF2_BACKENDS, has_hashlib_ssl)
        from passlib.utils.compat import PY3
        if PY3:
            self.assertIn('builtin-from-bytes', PBKDF2_BACKENDS)
        else:
            self.assertIn('builtin-unpack', PBKDF2_BACKENDS)

    def test_border(self):
        """test border cases"""

        def helper(secret=b'password', salt=b'salt', rounds=1, keylen=None, digest='sha1'):
            return pbkdf2_hmac(digest, secret, salt, rounds, keylen)
        helper()
        self.assertRaises(ValueError, helper, rounds=-1)
        self.assertRaises(ValueError, helper, rounds=0)
        self.assertRaises(TypeError, helper, rounds='x')
        helper(keylen=1)
        self.assertRaises(ValueError, helper, keylen=-1)
        self.assertRaises(ValueError, helper, keylen=0)
        self.assertRaises(OverflowError, helper, keylen=20 * (2 ** 32 - 1) + 1)
        self.assertRaises(TypeError, helper, keylen='x')
        self.assertRaises(TypeError, helper, salt=5)
        self.assertRaises(TypeError, helper, secret=5)
        self.assertRaises(ValueError, helper, digest='foo')
        self.assertRaises(TypeError, helper, digest=5)

    def test_default_keylen(self):
        """test keylen==None"""

        def helper(secret=b'password', salt=b'salt', rounds=1, keylen=None, digest='sha1'):
            return pbkdf2_hmac(digest, secret, salt, rounds, keylen)
        self.assertEqual(len(helper(digest='sha1')), 20)
        self.assertEqual(len(helper(digest='sha256')), 32)