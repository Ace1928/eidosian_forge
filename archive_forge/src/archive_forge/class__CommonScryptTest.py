from binascii import hexlify
import hashlib
import logging; log = logging.getLogger(__name__)
import struct
import warnings
from passlib import exc
from passlib.utils import getrandbytes
from passlib.utils.compat import PYPY, u, bascii_to_str
from passlib.utils.decor import classproperty
from passlib.tests.utils import TestCase, skipUnless, TEST_MODE, hb
from passlib.crypto import scrypt as scrypt_mod
class _CommonScryptTest(TestCase):
    """
    base class for testing various scrypt backends against same set of reference vectors.
    """

    @classproperty
    def descriptionPrefix(cls):
        return 'passlib.utils.scrypt.scrypt() <%s backend>' % cls.backend
    backend = None

    def setUp(self):
        assert self.backend
        scrypt_mod._set_backend(self.backend)
        super(_CommonScryptTest, self).setUp()
    reference_vectors = [('', '', 16, 1, 1, 64, hb('\n        77 d6 57 62 38 65 7b 20 3b 19 ca 42 c1 8a 04 97\n        f1 6b 48 44 e3 07 4a e8 df df fa 3f ed e2 14 42\n        fc d0 06 9d ed 09 48 f8 32 6a 75 3a 0f c8 1f 17\n        e8 d3 e0 fb 2e 0d 36 28 cf 35 e2 0c 38 d1 89 06\n        ')), ('password', 'NaCl', 1024, 8, 16, 64, hb('\n        fd ba be 1c 9d 34 72 00 78 56 e7 19 0d 01 e9 fe\n        7c 6a d7 cb c8 23 78 30 e7 73 76 63 4b 37 31 62\n        2e af 30 d9 2e 22 a3 88 6f f1 09 27 9d 98 30 da\n        c7 27 af b9 4a 83 ee 6d 83 60 cb df a2 cc 06 40\n        ')), ('pleaseletmein', 'SodiumChloride', 16384, 8, 1, 64, hb('\n        70 23 bd cb 3a fd 73 48 46 1c 06 cd 81 fd 38 eb\n        fd a8 fb ba 90 4f 8e 3e a9 b5 43 f6 54 5d a1 f2\n        d5 43 29 55 61 3f 0f cf 62 d4 97 05 24 2a 9a f9\n        e6 1e 85 dc 0d 65 1e 40 df cf 01 7b 45 57 58 87\n        ')), ('pleaseletmein', 'SodiumChloride', 1048576, 8, 1, 64, hb('\n        21 01 cb 9b 6a 51 1a ae ad db be 09 cf 70 f8 81\n        ec 56 8d 57 4a 2f fd 4d ab e5 ee 98 20 ad aa 47\n        8e 56 fd 8f 4b a5 d0 9f fa 1c 6d 92 7c 40 f4 c3\n        37 30 40 49 e8 a9 52 fb cb f4 5c 6f a7 7a 41 a4\n        '))]

    def test_reference_vectors(self):
        """reference vectors"""
        for secret, salt, n, r, p, keylen, result in self.reference_vectors:
            if n >= 1024 and TEST_MODE(max='default'):
                continue
            if n > 16384 and self.backend == 'builtin':
                continue
            log.debug('scrypt reference vector: %r %r n=%r r=%r p=%r', secret, salt, n, r, p)
            self.assertEqual(scrypt_mod.scrypt(secret, salt, n, r, p, keylen), result)
    _already_tested_others = None

    def test_other_backends(self):
        """compare output to other backends"""
        if self._already_tested_others:
            raise self.skipTest('already run under %r backend test' % self._already_tested_others)
        self._already_tested_others = self.backend
        rng = self.getRandom()
        orig = scrypt_mod.backend
        available = set((name for name in scrypt_mod.backend_values if scrypt_mod._has_backend(name)))
        scrypt_mod._set_backend(orig)
        available.discard(self.backend)
        if not available:
            raise self.skipTest('no other backends found')
        warnings.filterwarnings('ignore', '(?i)using builtin scrypt backend', category=exc.PasslibSecurityWarning)
        for _ in range(10):
            secret = getrandbytes(rng, rng.randint(0, 64))
            salt = getrandbytes(rng, rng.randint(0, 64))
            n = 1 << rng.randint(1, 10)
            r = rng.randint(1, 8)
            p = rng.randint(1, 3)
            ks = rng.randint(1, 64)
            previous = None
            backends = set()
            for name in available:
                scrypt_mod._set_backend(name)
                self.assertNotIn(scrypt_mod._scrypt, backends)
                backends.add(scrypt_mod._scrypt)
                result = hexstr(scrypt_mod.scrypt(secret, salt, n, r, p, ks))
                self.assertEqual(len(result), 2 * ks)
                if previous is not None:
                    self.assertEqual(result, previous, msg='%r output differs from others %r: %r' % (name, available, [secret, salt, n, r, p, ks]))

    def test_backend(self):
        """backend management"""
        scrypt_mod.backend = None
        scrypt_mod._scrypt = None
        self.assertRaises(TypeError, scrypt_mod.scrypt, 's', 's', 2, 2, 2, 16)
        scrypt_mod._set_backend(self.backend)
        self.assertEqual(scrypt_mod.backend, self.backend)
        scrypt_mod.scrypt('s', 's', 2, 2, 2, 16)
        self.assertRaises(ValueError, scrypt_mod._set_backend, 'xxx')
        self.assertEqual(scrypt_mod.backend, self.backend)

    def test_secret_param(self):
        """'secret' parameter"""

        def run_scrypt(secret):
            return hexstr(scrypt_mod.scrypt(secret, 'salt', 2, 2, 2, 16))
        TEXT = u('abcÞfg')
        self.assertEqual(run_scrypt(TEXT), '05717106997bfe0da42cf4779a2f8bd8')
        TEXT_UTF8 = b'abc\xc3\x9efg'
        self.assertEqual(run_scrypt(TEXT_UTF8), '05717106997bfe0da42cf4779a2f8bd8')
        TEXT_LATIN1 = b'abc\xdefg'
        self.assertEqual(run_scrypt(TEXT_LATIN1), '770825d10eeaaeaf98e8a3c40f9f441d')
        self.assertEqual(run_scrypt(''), 'ca1399e5fae5d3b9578dcd2b1faff6e2')
        self.assertRaises(TypeError, run_scrypt, None)
        self.assertRaises(TypeError, run_scrypt, 1)

    def test_salt_param(self):
        """'salt' parameter"""

        def run_scrypt(salt):
            return hexstr(scrypt_mod.scrypt('secret', salt, 2, 2, 2, 16))
        TEXT = u('abcÞfg')
        self.assertEqual(run_scrypt(TEXT), 'a748ec0f4613929e9e5f03d1ab741d88')
        TEXT_UTF8 = b'abc\xc3\x9efg'
        self.assertEqual(run_scrypt(TEXT_UTF8), 'a748ec0f4613929e9e5f03d1ab741d88')
        TEXT_LATIN1 = b'abc\xdefg'
        self.assertEqual(run_scrypt(TEXT_LATIN1), '91d056fb76fb6e9a7d1cdfffc0a16cd1')
        self.assertRaises(TypeError, run_scrypt, None)
        self.assertRaises(TypeError, run_scrypt, 1)

    def test_n_param(self):
        """'n' (rounds) parameter"""

        def run_scrypt(n):
            return hexstr(scrypt_mod.scrypt('secret', 'salt', n, 2, 2, 16))
        self.assertRaises(ValueError, run_scrypt, -1)
        self.assertRaises(ValueError, run_scrypt, 0)
        self.assertRaises(ValueError, run_scrypt, 1)
        self.assertEqual(run_scrypt(2), 'dacf2bca255e2870e6636fa8c8957a66')
        self.assertRaises(ValueError, run_scrypt, 3)
        self.assertRaises(ValueError, run_scrypt, 15)
        self.assertEqual(run_scrypt(16), '0272b8fc72bc54b1159340ed99425233')

    def test_r_param(self):
        """'r' (block size) parameter"""

        def run_scrypt(r, n=2, p=2):
            return hexstr(scrypt_mod.scrypt('secret', 'salt', n, r, p, 16))
        self.assertRaises(ValueError, run_scrypt, -1)
        self.assertRaises(ValueError, run_scrypt, 0)
        self.assertEqual(run_scrypt(1), '3d630447d9f065363b8a79b0b3670251')
        self.assertEqual(run_scrypt(2), 'dacf2bca255e2870e6636fa8c8957a66')
        self.assertEqual(run_scrypt(5), '114f05e985a903c27237b5578e763736')
        self.assertRaises(ValueError, run_scrypt, 1 << 30, p=1)
        self.assertRaises(ValueError, run_scrypt, (1 << 30) / 2, p=2)

    def test_p_param(self):
        """'p' (parallelism) parameter"""

        def run_scrypt(p, n=2, r=2):
            return hexstr(scrypt_mod.scrypt('secret', 'salt', n, r, p, 16))
        self.assertRaises(ValueError, run_scrypt, -1)
        self.assertRaises(ValueError, run_scrypt, 0)
        self.assertEqual(run_scrypt(1), 'f2960ea8b7d48231fcec1b89b784a6fa')
        self.assertEqual(run_scrypt(2), 'dacf2bca255e2870e6636fa8c8957a66')
        self.assertEqual(run_scrypt(5), '848a0eeb2b3543e7f543844d6ca79782')
        self.assertRaises(ValueError, run_scrypt, 1 << 30, r=1)
        self.assertRaises(ValueError, run_scrypt, (1 << 30) / 2, r=2)

    def test_keylen_param(self):
        """'keylen' parameter"""
        rng = self.getRandom()

        def run_scrypt(keylen):
            return hexstr(scrypt_mod.scrypt('secret', 'salt', 2, 2, 2, keylen))
        self.assertRaises(ValueError, run_scrypt, -1)
        self.assertRaises(ValueError, run_scrypt, 0)
        self.assertEqual(run_scrypt(1), 'da')
        ksize = rng.randint(1, 1 << 10)
        self.assertEqual(len(run_scrypt(ksize)), 2 * ksize)
        self.assertRaises(ValueError, run_scrypt, (2 ** 32 - 1) * 32 + 1)