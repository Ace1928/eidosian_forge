import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
class KangarooTwelveTest(unittest.TestCase):

    def test_length_encode(self):
        self.assertEqual(K12._length_encode(0), b'\x00')
        self.assertEqual(K12._length_encode(12), b'\x0c\x01')
        self.assertEqual(K12._length_encode(65538), b'\x01\x00\x02\x03')

    def test_new_positive(self):
        xof1 = K12.new()
        xof2 = K12.new(data=b('90'))
        xof3 = K12.new().update(b('90'))
        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))
        xof1 = K12.new()
        ref = xof1.read(10)
        xof2 = K12.new(custom=b(''))
        xof3 = K12.new(custom=b('foo'))
        self.assertEqual(ref, xof2.read(10))
        self.assertNotEqual(ref, xof3.read(10))
        xof1 = K12.new(custom=b('foo'))
        xof2 = K12.new(custom=b('foo'), data=b('90'))
        xof3 = K12.new(custom=b('foo')).update(b('90'))
        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = K12.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = K12.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = K12.new()
        self.assertRaises(TypeError, h.update, u'string')

    def test_digest(self):
        h = K12.new()
        digest = h.read(90)
        self.assertTrue(isinstance(digest, type(b('digest'))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = K12.new()
        mac.update(b('rrrr'))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b('ttt'))