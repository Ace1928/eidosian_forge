import json
import unittest
from binascii import unhexlify, hexlify
from .common import make_mac_tests
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import Poly1305
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
class Poly1305_Basic(object):

    @staticmethod
    def new(key, *data, **kwds):
        from Cryptodome.Hash.Poly1305 import Poly1305_MAC
        if len(data) == 1:
            msg = data[0]
        else:
            msg = None
        return Poly1305_MAC(key[:16], key[16:], msg)