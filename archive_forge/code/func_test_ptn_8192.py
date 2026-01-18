import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_8192(self):
    tv = '48 F2 56 F6 77 2F 9E DF B6 A8 B6 61 EC 92 DC 93\n        B9 5E BD 05 A0 8A 17 B3 9A E3 49 08 70 C9 26 C3'
    btv = txt2bin(tv)
    res = K12.new(data=ptn(8192)).read(32)
    self.assertEqual(res, btv)