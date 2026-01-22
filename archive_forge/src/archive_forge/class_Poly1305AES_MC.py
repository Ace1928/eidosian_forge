import json
import unittest
from binascii import unhexlify, hexlify
from .common import make_mac_tests
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import Poly1305
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
class Poly1305AES_MC(unittest.TestCase):

    def runTest(self):
        tag = unhexlify(b'fb447350c4e868c52ac3275cf9d4327e')
        msg = b''
        for msg_len in range(5000 + 1):
            key = tag + strxor_c(tag, 255)
            nonce = tag[::-1]
            if msg_len > 0:
                msg = msg + tobytes(tag[0])
            auth = Poly1305.new(key=key, nonce=nonce, cipher=AES, data=msg)
            tag = auth.digest()
        self.assertEqual('CDFA436DDD629C7DC20E1128530BAED2', auth.hexdigest().upper())