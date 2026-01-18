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