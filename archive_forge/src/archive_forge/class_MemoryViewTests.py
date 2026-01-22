import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import tostr, tobytes
from Cryptodome.Hash import (HMAC, MD5, SHA1, SHA256,
class MemoryViewTests(unittest.TestCase):

    def runTest(self):
        key = b'0' * 16
        data = b'\x00\x01\x02'

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))
        for get_mv in (get_mv_ro, get_mv_rw):
            key_mv = get_mv(key)
            data_mv = get_mv(data)
            h1 = HMAC.new(key, data)
            h2 = HMAC.new(key_mv, data_mv)
            if not data_mv.readonly:
                key_mv[:1] = b'\xff'
                data_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())
            data_mv = get_mv(data)
            h1 = HMAC.new(key)
            h2 = HMAC.new(key)
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())