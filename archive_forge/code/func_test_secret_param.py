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