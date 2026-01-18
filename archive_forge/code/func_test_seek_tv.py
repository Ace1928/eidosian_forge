import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
def test_seek_tv(self):
    key = bchr(0) + bchr(255) + bchr(0) * 30
    nonce = bchr(0) * 8
    cipher = ChaCha20.new(key=key, nonce=nonce)
    cipher.seek(64 * 2)
    expected_key_stream = unhexlify(b('72d54dfbf12ec44b362692df94137f328fea8da73990265ec1bbbea1ae9af0ca13b25aa26cb4a648cb9b9d1be65b2c0924a66c54d545ec1b7374f4872e99f096'))
    ct = cipher.encrypt(bchr(0) * len(expected_key_stream))
    self.assertEqual(expected_key_stream, ct)