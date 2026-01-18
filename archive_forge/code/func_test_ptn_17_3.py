import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ptn_17_3(self):
    tv = 'C7 4E BC 91 9A 5B 3B 0D D1 22 81 85 BA 02 D2 9E\n        F4 42 D6 9D 3D 42 76 A9 3E FE 0B F9 A1 6A 7D C0\n        CD 4E AB AD AB 8C D7 A5 ED D9 66 95 F5 D3 60 AB\n        E0 9E 2C 65 11 A3 EC 39 7D A3 B7 6B 9E 16 74 FB'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=ptn(17 ** 3)).read(64)
    self.assertEqual(res, btv)