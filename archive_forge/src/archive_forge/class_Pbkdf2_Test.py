from __future__ import with_statement
import hashlib
import warnings
from passlib.utils.compat import u, JYTHON
from passlib.tests.utils import TestCase, hb
class Pbkdf2_Test(TestCase):
    """test pbkdf2() support"""
    descriptionPrefix = 'passlib.utils.pbkdf2.pbkdf2()'
    pbkdf2_test_vectors = [(hb('cdedb5281bb2f801565a1122b2563515'), b'password', b'ATHENA.MIT.EDUraeburn', 1, 16), (hb('01dbee7f4a9e243e988b62c73cda935d'), b'password', b'ATHENA.MIT.EDUraeburn', 2, 16), (hb('01dbee7f4a9e243e988b62c73cda935da05378b93244ec8f48a99e61ad799d86'), b'password', b'ATHENA.MIT.EDUraeburn', 2, 32), (hb('5c08eb61fdf71e4e4ec3cf6ba1f5512ba7e52ddbc5e5142f708a31e2e62b1e13'), b'password', b'ATHENA.MIT.EDUraeburn', 1200, 32), (hb('d1daa78615f287e6a1c8b120d7062a493f98d203e6be49a6adf4fa574b6e64ee'), b'password', b'\x124VxxV4\x12', 5, 32), (hb('139c30c0966bc32ba55fdbf212530ac9c5ec59f1a452f5cc9ad940fea0598ed1'), b'X' * 64, b'pass phrase equals block size', 1200, 32), (hb('9ccad6d468770cd51b10e6a68721be611a8b4d282601db3b36be9246915ec82a'), b'X' * 65, b'pass phrase exceeds block size', 1200, 32), (hb('0c60c80f961f0e71f3a9b524af6012062fe037a6'), b'password', b'salt', 1, 20), (hb('ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957'), b'password', b'salt', 2, 20), (hb('4b007901b765489abead49d926f721d065a429c1'), b'password', b'salt', 4096, 20), (hb('3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038'), b'passwordPASSWORDpassword', b'saltSALTsaltSALTsaltSALTsaltSALTsalt', 4096, 25), (hb('56fa6aa75548099dcc37d7f03425e0c3'), b'pass\x00word', b'sa\x00lt', 4096, 16), (hb('887CFF169EA8335235D8004242AA7D6187A41E3187DF0CE14E256D85ED97A97357AAA8FF0A3871AB9EEFF458392F462F495487387F685B7472FC6C29E293F0A0'), b'hello', hb('9290F727ED06C38BA4549EF7DE25CF5642659211B7FC076F2D28FEFD71784BB8D8F6FB244A8CC5C06240631B97008565A120764C0EE9C2CB0073994D79080136'), 10000, 64, 'hmac-sha512'), (hb('e248fb6b13365146f8ac6307cc222812'), b'secret', b'salt', 10, 16, 'hmac-sha1'), (hb('e248fb6b13365146f8ac6307cc2228127872da6d'), b'secret', b'salt', 10, None, 'hmac-sha1')]

    def setUp(self):
        super(Pbkdf2_Test, self).setUp()
        warnings.filterwarnings('ignore', '.*passlib.utils.pbkdf2.*deprecated', DeprecationWarning)

    def test_known(self):
        """test reference vectors"""
        from passlib.utils.pbkdf2 import pbkdf2
        for row in self.pbkdf2_test_vectors:
            correct, secret, salt, rounds, keylen = row[:5]
            prf = row[5] if len(row) == 6 else 'hmac-sha1'
            result = pbkdf2(secret, salt, rounds, keylen, prf)
            self.assertEqual(result, correct)

    def test_border(self):
        """test border cases"""
        from passlib.utils.pbkdf2 import pbkdf2

        def helper(secret=b'password', salt=b'salt', rounds=1, keylen=None, prf='hmac-sha1'):
            return pbkdf2(secret, salt, rounds, keylen, prf)
        helper()
        self.assertRaises(ValueError, helper, rounds=-1)
        self.assertRaises(ValueError, helper, rounds=0)
        self.assertRaises(TypeError, helper, rounds='x')
        self.assertRaises(ValueError, helper, keylen=-1)
        self.assertRaises(ValueError, helper, keylen=0)
        helper(keylen=1)
        self.assertRaises(OverflowError, helper, keylen=20 * (2 ** 32 - 1) + 1)
        self.assertRaises(TypeError, helper, keylen='x')
        self.assertRaises(TypeError, helper, salt=5)
        self.assertRaises(TypeError, helper, secret=5)
        self.assertRaises(ValueError, helper, prf='hmac-foo')
        self.assertRaises(NotImplementedError, helper, prf='foo')
        self.assertRaises(TypeError, helper, prf=5)

    def test_default_keylen(self):
        """test keylen==None"""
        from passlib.utils.pbkdf2 import pbkdf2

        def helper(secret=b'password', salt=b'salt', rounds=1, keylen=None, prf='hmac-sha1'):
            return pbkdf2(secret, salt, rounds, keylen, prf)
        self.assertEqual(len(helper(prf='hmac-sha1')), 20)
        self.assertEqual(len(helper(prf='hmac-sha256')), 32)

    def test_custom_prf(self):
        """test custom prf function"""
        from passlib.utils.pbkdf2 import pbkdf2

        def prf(key, msg):
            return hashlib.md5(key + msg + b'fooey').digest()
        self.assertRaises(NotImplementedError, pbkdf2, b'secret', b'salt', 1000, 20, prf)