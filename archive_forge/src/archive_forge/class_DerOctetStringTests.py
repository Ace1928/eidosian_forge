import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerOctetStringTests(unittest.TestCase):

    def testInit1(self):
        der = DerOctetString(b('ÿ'))
        self.assertEqual(der.encode(), b('\x04\x01ÿ'))

    def testEncode1(self):
        der = DerOctetString()
        self.assertEqual(der.encode(), b('\x04\x00'))
        der.payload = b('\x01\x02')
        self.assertEqual(der.encode(), b('\x04\x02\x01\x02'))

    def testDecode1(self):
        der = DerOctetString()
        der.decode(b('\x04\x00'))
        self.assertEqual(der.payload, b(''))
        der.decode(b('\x04\x02\x01\x02'))
        self.assertEqual(der.payload, b('\x01\x02'))

    def testDecode2(self):
        der = DerOctetString()
        self.assertEqual(der, der.decode(b('\x04\x00')))

    def testErrDecode1(self):
        der = DerOctetString()
        self.assertRaises(ValueError, der.decode, b('\x04\x01\x01ÿ'))