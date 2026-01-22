import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerBooleanTests(unittest.TestCase):

    def testEncode1(self):
        der = DerBoolean(False)
        self.assertEqual(der.encode(), b'\x01\x01\x00')

    def testEncode2(self):
        der = DerBoolean(True)
        self.assertEqual(der.encode(), b'\x01\x01\xff')

    def testEncode3(self):
        der = DerBoolean(False, implicit=18)
        self.assertEqual(der.encode(), b'\x92\x01\x00')

    def testEncode4(self):
        der = DerBoolean(False, explicit=5)
        self.assertEqual(der.encode(), b'\xa5\x03\x01\x01\x00')

    def testDecode1(self):
        der = DerBoolean()
        der.decode(b'\x01\x01\x00')
        self.assertEqual(der.value, False)

    def testDecode2(self):
        der = DerBoolean()
        der.decode(b'\x01\x01\xff')
        self.assertEqual(der.value, True)

    def testDecode3(self):
        der = DerBoolean(implicit=18)
        der.decode(b'\x92\x01\x00')
        self.assertEqual(der.value, False)

    def testDecode4(self):
        der = DerBoolean(explicit=5)
        der.decode(b'\xa5\x03\x01\x01\x00')
        self.assertEqual(der.value, False)

    def testErrorDecode1(self):
        der = DerBoolean()
        self.assertRaises(ValueError, der.decode, b'\x02\x01\x00')

    def testErrorDecode2(self):
        der = DerBoolean()
        self.assertRaises(ValueError, der.decode, b'\x01\x01\x00\xff')