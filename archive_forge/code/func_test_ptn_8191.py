import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_8191(self):
    tv = '1B 57 76 36 F7 23 64 3E 99 0C C7 D6 A6 59 83 74\n        36 FD 6A 10 36 26 60 0E B8 30 1C D1 DB E5 53 D6'
    btv = txt2bin(tv)
    res = K12.new(data=ptn(8191)).read(32)
    self.assertEqual(res, btv)