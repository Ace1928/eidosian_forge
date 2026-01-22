import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
class DerNullTests(unittest.TestCase):

    def testEncode1(self):
        der = DerNull()
        self.assertEqual(der.encode(), b('\x05\x00'))

    def testDecode1(self):
        der = DerNull()
        self.assertEqual(der, der.decode(b('\x05\x00')))