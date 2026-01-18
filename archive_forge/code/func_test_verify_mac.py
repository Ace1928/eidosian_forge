import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Hash import CMAC
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def test_verify_mac(self, tv):
    self._id = 'Wycheproof MAC verification Test #' + str(tv.id)
    try:
        mac = CMAC.new(tv.key, tv.msg, ciphermod=AES, mac_len=tv.tag_size)
    except ValueError as e:
        if len(tv.key) not in (16, 24, 32) and 'key length' in str(e):
            return
        raise e
    try:
        mac.verify(tv.tag)
    except ValueError:
        assert not tv.valid
    else:
        assert tv.valid
        self.warn(tv)