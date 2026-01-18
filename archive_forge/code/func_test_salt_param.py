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