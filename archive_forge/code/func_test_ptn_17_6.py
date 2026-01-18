import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ptn_17_6(self):
    tv = '9E 11 BC 59 C2 4E 73 99 3C 14 84 EC 66 35 8E F7\n        1D B7 4A EF D8 4E 12 3F 78 00 BA 9C 48 53 E0 2C\n        FE 70 1D 9E 6B B7 65 A3 04 F0 DC 34 A4 EE 3B A8\n        2C 41 0F 0D A7 0E 86 BF BD 90 EA 87 7C 2D 61 04'
    btv = txt2bin(tv)
    data = ptn(17 ** 6)
    res = TurboSHAKE256.new(data=data).read(64)
    self.assertEqual(res, btv)