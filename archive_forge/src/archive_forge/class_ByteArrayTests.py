import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import tostr, tobytes
from Cryptodome.Hash import (HMAC, MD5, SHA1, SHA256,
class ByteArrayTests(unittest.TestCase):

    def runTest(self):
        key = b'0' * 16
        data = b'\x00\x01\x02'
        key_ba = bytearray(key)
        data_ba = bytearray(data)
        h1 = HMAC.new(key, data)
        h2 = HMAC.new(key_ba, data_ba)
        key_ba[:1] = b'\xff'
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())
        key_ba = bytearray(key)
        data_ba = bytearray(data)
        h1 = HMAC.new(key)
        h2 = HMAC.new(key)
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())