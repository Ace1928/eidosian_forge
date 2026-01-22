import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2Test(unittest.TestCase):

    def test_new_positive(self):
        h = self.BLAKE2.new(digest_bits=self.max_bits)
        for new_func in (self.BLAKE2.new, h.new):
            for dbits in range(8, self.max_bits + 1, 8):
                hobj = new_func(digest_bits=dbits)
                self.assertEqual(hobj.digest_size, dbits // 8)
            for dbytes in range(1, self.max_bytes + 1):
                hobj = new_func(digest_bytes=dbytes)
                self.assertEqual(hobj.digest_size, dbytes)
            digest1 = new_func(data=b'\x90', digest_bytes=self.max_bytes).digest()
            digest2 = new_func(digest_bytes=self.max_bytes).update(b'\x90').digest()
            self.assertEqual(digest1, digest2)
            new_func(data=b'A', key=b'5', digest_bytes=self.max_bytes)
        hobj = h.new()
        self.assertEqual(hobj.digest_size, self.max_bytes)

    def test_new_negative(self):
        h = self.BLAKE2.new(digest_bits=self.max_bits)
        for new_func in (self.BLAKE2.new, h.new):
            self.assertRaises(TypeError, new_func, digest_bytes=self.max_bytes, digest_bits=self.max_bits)
            self.assertRaises(ValueError, new_func, digest_bytes=0)
            self.assertRaises(ValueError, new_func, digest_bytes=self.max_bytes + 1)
            self.assertRaises(ValueError, new_func, digest_bits=7)
            self.assertRaises(ValueError, new_func, digest_bits=15)
            self.assertRaises(ValueError, new_func, digest_bits=self.max_bits + 1)
            self.assertRaises(TypeError, new_func, digest_bytes=self.max_bytes, key=u'string')
            self.assertRaises(TypeError, new_func, digest_bytes=self.max_bytes, data=u'string')

    def test_default_digest_size(self):
        digest = self.BLAKE2.new(data=b'abc').digest()
        self.assertEqual(len(digest), self.max_bytes)

    def test_update(self):
        pieces = [b'\n' * 200, b'\x14' * 300]
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        h.update(pieces[0]).update(pieces[1])
        digest = h.digest()
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.digest(), digest)

    def test_update_negative(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        self.assertRaises(TypeError, h.update, u'string')

    def test_digest(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        digest = h.digest()
        self.assertEqual(h.digest(), digest)
        self.assertTrue(isinstance(digest, type(b'digest')))

    def test_update_after_digest(self):
        msg = b'rrrrttt'
        h = self.BLAKE2.new(digest_bits=256, data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = self.BLAKE2.new(digest_bits=256, data=msg).digest()
        h = self.BLAKE2.new(digest_bits=256, data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)

    def test_hex_digest(self):
        mac = self.BLAKE2.new(digest_bits=self.max_bits)
        digest = mac.digest()
        hexdigest = mac.hexdigest()
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        self.assertEqual(mac.hexdigest(), hexdigest)
        self.assertTrue(isinstance(hexdigest, type('digest')))

    def test_verify(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes, key=b'4')
        mac = h.digest()
        h.verify(mac)
        wrong_mac = strxor_c(mac, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)

    def test_hexverify(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes, key=b'4')
        mac = h.hexdigest()
        h.hexverify(mac)
        self.assertRaises(ValueError, h.hexverify, '4556')

    def test_oid(self):
        prefix = '1.3.6.1.4.1.1722.12.2.' + self.oid_variant + '.'
        for digest_bits in self.digest_bits_oid:
            h = self.BLAKE2.new(digest_bits=digest_bits)
            self.assertEqual(h.oid, prefix + str(digest_bits // 8))
            h = self.BLAKE2.new(digest_bits=digest_bits, key=b'secret')
            self.assertRaises(AttributeError, lambda: h.oid)
        for digest_bits in (8, self.max_bits):
            if digest_bits in self.digest_bits_oid:
                continue
            self.assertRaises(AttributeError, lambda: h.oid)

    def test_bytearray(self):
        key = b'0' * 16
        data = b'\x00\x01\x02'
        key_ba = bytearray(key)
        data_ba = bytearray(data)
        h1 = self.BLAKE2.new(data=data, key=key)
        h2 = self.BLAKE2.new(data=data_ba, key=key_ba)
        key_ba[:1] = b'\xff'
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())
        data_ba = bytearray(data)
        h1 = self.BLAKE2.new()
        h2 = self.BLAKE2.new()
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):
        key = b'0' * 16
        data = b'\x00\x01\x02'

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))
        for get_mv in (get_mv_ro, get_mv_rw):
            key_mv = get_mv(key)
            data_mv = get_mv(data)
            h1 = self.BLAKE2.new(data=data, key=key)
            h2 = self.BLAKE2.new(data=data_mv, key=key_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xff'
                key_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())
            data_mv = get_mv(data)
            h1 = self.BLAKE2.new()
            h2 = self.BLAKE2.new()
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())