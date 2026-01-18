import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ptn_1(self):
    tv = '3E 17 12 F9 28 F8 EA F1 05 46 32 B2 AA 0A 24 6E\n        D8 B0 C3 78 72 8F 60 BC 97 04 10 15 5C 28 82 0E\n        90 CC 90 D8 A3 00 6A A2 37 2C 5C 5E A1 76 B0 68\n        2B F2 2B AE 74 67 AC 94 F7 4D 43 D3 9B 04 82 E2'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=ptn(1)).read(64)
    self.assertEqual(res, btv)