import unittest
from binascii import unhexlify as uh
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.Padding import pad, unpad
class PKCS7_Tests(unittest.TestCase):

    def test1(self):
        padded = pad(b(''), 4)
        self.assertTrue(padded == uh(b('04040404')))
        padded = pad(b(''), 4, 'pkcs7')
        self.assertTrue(padded == uh(b('04040404')))
        back = unpad(padded, 4)
        self.assertTrue(back == b(''))

    def test2(self):
        padded = pad(uh(b('12345678')), 4)
        self.assertTrue(padded == uh(b('1234567804040404')))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b('12345678')))

    def test3(self):
        padded = pad(uh(b('123456')), 4)
        self.assertTrue(padded == uh(b('12345601')))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b('123456')))

    def test4(self):
        padded = pad(uh(b('1234567890')), 4)
        self.assertTrue(padded == uh(b('1234567890030303')))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b('1234567890')))

    def testn1(self):
        self.assertRaises(ValueError, pad, uh(b('12')), 4, 'pkcs8')

    def testn2(self):
        self.assertRaises(ValueError, unpad, b('\x00\x00\x00'), 4)
        self.assertRaises(ValueError, unpad, b(''), 4)

    def testn3(self):
        self.assertRaises(ValueError, unpad, b('123456\x02'), 4)
        self.assertRaises(ValueError, unpad, b('123456\x00'), 4)
        self.assertRaises(ValueError, unpad, b('123456\x05\x05\x05\x05\x05'), 4)