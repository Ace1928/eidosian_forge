import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KMAC128, KMAC256
class KMACTest(unittest.TestCase):

    def new(self, *args, **kwargs):
        return self.KMAC.new(*args, key=b'X' * (self.minimum_key_bits // 8), **kwargs)

    def test_new_positive(self):
        key = b'X' * 32
        h = self.new()
        for new_func in (self.KMAC.new, h.new):
            for dbytes in range(self.minimum_bytes, 128 + 1):
                hobj = new_func(key=key, mac_len=dbytes)
                self.assertEqual(hobj.digest_size, dbytes)
            digest1 = new_func(key=key, data=b'\x90').digest()
            digest2 = new_func(key=key).update(b'\x90').digest()
            self.assertEqual(digest1, digest2)
            new_func(data=b'A', key=key, custom=b'g')
        hobj = h.new(key=key)
        self.assertEqual(hobj.digest_size, self.default_bytes)

    def test_new_negative(self):
        h = self.new()
        for new_func in (self.KMAC.new, h.new):
            self.assertRaises(ValueError, new_func, key=b'X' * 32, mac_len=0)
            self.assertRaises(ValueError, new_func, key=b'X' * 32, mac_len=self.minimum_bytes - 1)
            self.assertRaises(TypeError, new_func, key=u'string')
            self.assertRaises(TypeError, new_func, data=u'string')

    def test_default_digest_size(self):
        digest = self.new(data=b'abc').digest()
        self.assertEqual(len(digest), self.default_bytes)

    def test_update(self):
        pieces = [b'\n' * 200, b'\x14' * 300]
        h = self.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.digest()
        h = self.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.digest(), digest)

    def test_update_negative(self):
        h = self.new()
        self.assertRaises(TypeError, h.update, u'string')

    def test_digest(self):
        h = self.new()
        digest = h.digest()
        self.assertEqual(h.digest(), digest)
        self.assertTrue(isinstance(digest, type(b'digest')))

    def test_update_after_digest(self):
        msg = b'rrrrttt'
        h = self.new(mac_len=32, data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, dig1)

    def test_hex_digest(self):
        mac = self.new()
        digest = mac.digest()
        hexdigest = mac.hexdigest()
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        self.assertEqual(mac.hexdigest(), hexdigest)
        self.assertTrue(isinstance(hexdigest, type('digest')))

    def test_verify(self):
        h = self.new()
        mac = h.digest()
        h.verify(mac)
        wrong_mac = strxor_c(mac, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)

    def test_hexverify(self):
        h = self.new()
        mac = h.hexdigest()
        h.hexverify(mac)
        self.assertRaises(ValueError, h.hexverify, '4556')

    def test_oid(self):
        oid = '2.16.840.1.101.3.4.2.' + self.oid_variant
        h = self.new()
        self.assertEqual(h.oid, oid)

    def test_bytearray(self):
        key = b'0' * 32
        data = b'\x00\x01\x02'
        key_ba = bytearray(key)
        data_ba = bytearray(data)
        h1 = self.KMAC.new(data=data, key=key)
        h2 = self.KMAC.new(data=data_ba, key=key_ba)
        key_ba[:1] = b'\xff'
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())
        data_ba = bytearray(data)
        h1 = self.new()
        h2 = self.new()
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xff'
        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):
        key = b'0' * 32
        data = b'\x00\x01\x02'

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))
        for get_mv in (get_mv_ro, get_mv_rw):
            key_mv = get_mv(key)
            data_mv = get_mv(data)
            h1 = self.KMAC.new(data=data, key=key)
            h2 = self.KMAC.new(data=data_mv, key=key_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xff'
                key_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())
            data_mv = get_mv(data)
            h1 = self.new()
            h2 = self.new()
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xff'
            self.assertEqual(h1.digest(), h2.digest())