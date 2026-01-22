import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerObjectIdTests(unittest.TestCase):

    def testInit1(self):
        der = DerObjectId('1.1')
        self.assertEqual(der.encode(), b'\x06\x01)')

    def testEncode1(self):
        der = DerObjectId('1.2.840.113549.1.1.1')
        self.assertEqual(der.encode(), b'\x06\t*\x86H\x86\xf7\r\x01\x01\x01')
        der = DerObjectId()
        der.value = '1.2.840.113549.1.1.1'
        self.assertEqual(der.encode(), b'\x06\t*\x86H\x86\xf7\r\x01\x01\x01')
        der = DerObjectId('2.999.1234')
        self.assertEqual(der.encode(), b'\x06\x04\x887\x89R')

    def testEncode2(self):
        der = DerObjectId('3.4')
        self.assertRaises(ValueError, der.encode)
        der = DerObjectId('1.40')
        self.assertRaises(ValueError, der.encode)

    def testDecode1(self):
        der = DerObjectId()
        der.decode(b'\x06\t*\x86H\x86\xf7\r\x01\x01\x01')
        self.assertEqual(der.value, '1.2.840.113549.1.1.1')

    def testDecode2(self):
        der = DerObjectId()
        self.assertEqual(der, der.decode(b'\x06\t*\x86H\x86\xf7\r\x01\x01\x01'))

    def testDecode3(self):
        der = DerObjectId()
        der.decode(b'\x06\t*\x86H\x86\xf7\r\x01\x00\x01')
        self.assertEqual(der.value, '1.2.840.113549.1.0.1')

    def testDecode4(self):
        der = DerObjectId()
        der.decode(b'\x06\x04\x887\x89R')
        self.assertEqual(der.value, '2.999.1234')