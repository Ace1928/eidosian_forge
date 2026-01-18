import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_c_41_2(self):
    tv = 'C3 89 E5 00 9A E5 71 20 85 4C 2E 8C 64 67 0A C0\n        13 58 CF 4C 1B AF 89 44 7A 72 42 34 DC 7C ED 74'
    btv = txt2bin(tv)
    custom = ptn(41 ** 2)
    res = K12.new(data=b'\xff' * 3, custom=custom).read(32)
    self.assertEqual(res, btv)