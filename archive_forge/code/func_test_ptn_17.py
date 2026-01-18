import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ptn_17(self):
    tv = 'B3 BA B0 30 0E 6A 19 1F BE 61 37 93 98 35 92 35\n        78 79 4E A5 48 43 F5 01 10 90 FA 2F 37 80 A9 E5\n        CB 22 C5 9D 78 B4 0A 0F BF F9 E6 72 C0 FB E0 97\n        0B D2 C8 45 09 1C 60 44 D6 87 05 4D A5 D8 E9 C7'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=ptn(17)).read(64)
    self.assertEqual(res, btv)