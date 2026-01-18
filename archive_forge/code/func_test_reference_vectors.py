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
def test_reference_vectors(self):
    """reference vectors"""
    for secret, salt, n, r, p, keylen, result in self.reference_vectors:
        if n >= 1024 and TEST_MODE(max='default'):
            continue
        if n > 16384 and self.backend == 'builtin':
            continue
        log.debug('scrypt reference vector: %r %r n=%r r=%r p=%r', secret, salt, n, r, p)
        self.assertEqual(scrypt_mod.scrypt(secret, salt, n, r, p, keylen), result)