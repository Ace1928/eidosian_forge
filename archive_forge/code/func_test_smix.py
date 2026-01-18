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
def test_smix(self):
    """smix()"""
    from passlib.crypto.scrypt._builtin import ScryptEngine
    rng = self.getRandom()
    input = hb('\n            f7 ce 0b 65 3d 2d 72 a4 10 8c f5 ab e9 12 ff dd\n            77 76 16 db bb 27 a7 0e 82 04 f3 ae 2d 0f 6f ad\n            89 f6 8f 48 11 d1 e8 7b cc 3b d7 40 0a 9f fd 29\n            09 4f 01 84 63 95 74 f3 9a e5 a1 31 52 17 bc d7\n            89 49 91 44 72 13 bb 22 6c 25 b5 4d a8 63 70 fb\n            cd 98 43 80 37 46 66 bb 8f fc b5 bf 40 c2 54 b0\n            67 d2 7c 51 ce 4a d5 fe d8 29 c9 0b 50 5a 57 1b\n            7f 4d 1c ad 6a 52 3c da 77 0e 67 bc ea af 7e 89\n            ')
    output = hb('\n            79 cc c1 93 62 9d eb ca 04 7f 0b 70 60 4b f6 b6\n            2c e3 dd 4a 96 26 e3 55 fa fc 61 98 e6 ea 2b 46\n            d5 84 13 67 3b 99 b0 29 d6 65 c3 57 60 1f b4 26\n            a0 b2 f4 bb a2 00 ee 9f 0a 43 d1 9b 57 1a 9c 71\n            ef 11 42 e6 5d 5a 26 6f dd ca 83 2c e5 9f aa 7c\n            ac 0b 9c f1 be 2b ff ca 30 0d 01 ee 38 76 19 c4\n            ae 12 fd 44 38 f2 03 a0 e4 e1 c4 7e c3 14 86 1f\n            4e 90 87 cb 33 39 6a 68 73 e8 f9 d2 53 9a 4b 8e\n            ')
    engine = ScryptEngine(n=16, r=1, p=rng.randint(1, 1023))
    self.assertEqual(engine.smix(input), output)