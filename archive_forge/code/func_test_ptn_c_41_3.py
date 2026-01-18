import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_c_41_3(self):
    tv = '75 D2 F8 6A 2E 64 45 66 72 6B 4F BC FC 56 57 B9\n        DB CF 07 0C 7B 0D CA 06 45 0A B2 91 D7 44 3B CF'
    btv = txt2bin(tv)
    custom = ptn(41 ** 3)
    res = K12.new(data=b'\xff' * 7, custom=custom).read(32)
    self.assertEqual(res, btv)