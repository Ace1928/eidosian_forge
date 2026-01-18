import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_zero_1(self):
    tv = '36 7A 32 9D AF EA 87 1C 78 02 EC 67 F9 05 AE 13\n        C5 76 95 DC 2C 66 63 C6 10 35 F5 9A 18 F8 E7 DB\n        11 ED C0 E1 2E 91 EA 60 EB 6B 32 DF 06 DD 7F 00\n        2F BA FA BB 6E 13 EC 1C C2 0D 99 55 47 60 0D B0'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new().read(64)
    self.assertEqual(res, btv)