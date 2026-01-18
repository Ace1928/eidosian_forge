import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_c_1(self):
    tv = 'FA B6 58 DB 63 E9 4A 24 61 88 BF 7A F6 9A 13 30\n        45 F4 6E E9 84 C5 6E 3C 33 28 CA AF 1A A1 A5 83'
    btv = txt2bin(tv)
    custom = ptn(1)
    res = K12.new(custom=custom).read(32)
    self.assertEqual(res, btv)