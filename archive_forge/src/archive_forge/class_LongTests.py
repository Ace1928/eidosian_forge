import math
import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes
class LongTests(unittest.TestCase):

    def test1(self):
        self.assertEqual(long_to_bytes(0), b'\x00')
        self.assertEqual(long_to_bytes(1), b'\x01')
        self.assertEqual(long_to_bytes(256), b'\x01\x00')
        self.assertEqual(long_to_bytes(1095216660480), b'\xff\x00\x00\x00\x00')
        self.assertEqual(long_to_bytes(1095216660480), b'\xff\x00\x00\x00\x00')
        self.assertEqual(long_to_bytes(1234605616436508552), b'\x11"3DUfw\x88')
        self.assertEqual(long_to_bytes(316059037807746189465), b'\x11"3DUfw\x88\x99')

    def test2(self):
        self.assertEqual(long_to_bytes(0, 1), b'\x00')
        self.assertEqual(long_to_bytes(0, 2), b'\x00\x00')
        self.assertEqual(long_to_bytes(1, 3), b'\x00\x00\x01')
        self.assertEqual(long_to_bytes(65535, 2), b'\xff\xff')
        self.assertEqual(long_to_bytes(65536, 2), b'\x00\x01\x00\x00')
        self.assertEqual(long_to_bytes(256, 1), b'\x01\x00')
        self.assertEqual(long_to_bytes(1095216660481, 6), b'\x00\xff\x00\x00\x00\x01')
        self.assertEqual(long_to_bytes(1095216660481, 8), b'\x00\x00\x00\xff\x00\x00\x00\x01')
        self.assertEqual(long_to_bytes(1095216660481, 10), b'\x00\x00\x00\x00\x00\xff\x00\x00\x00\x01')
        self.assertEqual(long_to_bytes(1095216660481, 11), b'\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\x01')

    def test_err1(self):
        self.assertRaises(ValueError, long_to_bytes, -1)