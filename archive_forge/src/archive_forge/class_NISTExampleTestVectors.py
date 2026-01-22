import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
class NISTExampleTestVectors(unittest.TestCase):
    test_data = [(('00 01 02', '10 11 12 13 14 15'), '', 'C5 D8 78 6C 1A FB 9B 82 11 1A B3 4B 65 B2 C0 048F A6 4E 6D 48 E2 63 26 4C E1 70 7D 3F FC 8E D1', 'KMAC128 Sample #1 NIST', TupleHash128), (('00 01 02', '10 11 12 13 14 15'), 'My Tuple App', '75 CD B2 0F F4 DB 11 54 E8 41 D7 58 E2 41 60 C54B AE 86 EB 8C 13 E7 F5 F4 0E B3 55 88 E9 6D FB', 'KMAC128 Sample #2 NIST', TupleHash128), (('00 01 02', '10 11 12 13 14 15', '20 21 22 23 24 25 26 27 28'), 'My Tuple App', 'E6 0F 20 2C 89 A2 63 1E DA 8D 4C 58 8C A5 FD 07F3 9E 51 51 99 8D EC CF 97 3A DB 38 04 BB 6E 84', 'KMAC128 Sample #3 NIST', TupleHash128), (('00 01 02', '10 11 12 13 14 15'), '', 'CF B7 05 8C AC A5 E6 68 F8 1A 12 A2 0A 21 95 CE97 A9 25 F1 DB A3 E7 44 9A 56 F8 22 01 EC 60 7311 AC 26 96 B1 AB 5E A2 35 2D F1 42 3B DE 7B D4BB 78 C9 AE D1 A8 53 C7 86 72 F9 EB 23 BB E1 94', 'KMAC256 Sample #4 NIST', TupleHash256), (('00 01 02', '10 11 12 13 14 15'), 'My Tuple App', '14 7C 21 91 D5 ED 7E FD 98 DB D9 6D 7A B5 A1 1692 57 6F 5F E2 A5 06 5F 3E 33 DE 6B BA 9F 3A A1C4 E9 A0 68 A2 89 C6 1C 95 AA B3 0A EE 1E 41 0B0B 60 7D E3 62 0E 24 A4 E3 BF 98 52 A1 D4 36 7E', 'KMAC256 Sample #5 NIST', TupleHash256), (('00 01 02', '10 11 12 13 14 15', '20 21 22 23 24 25 26 27 28'), 'My Tuple App', '45 00 0B E6 3F 9B 6B FD 89 F5 47 17 67 0F 69 A9BC 76 35 91 A4 F0 5C 50 D6 88 91 A7 44 BC C6 E7D6 D5 B5 E8 2C 01 8D A9 99 ED 35 B0 BB 49 C9 678E 52 6A BD 8E 85 C1 3E D2 54 02 1D B9 E7 90 CE', 'KMAC256 Sample #6 NIST', TupleHash256)]

    def setUp(self):
        td = []
        for tv_in in self.test_data:
            tv_out = [None] * len(tv_in)
            tv_out[0] = []
            for string in tv_in[0]:
                tv_out[0].append(unhexlify(string.replace(' ', '')))
            tv_out[1] = tobytes(tv_in[1])
            tv_out[2] = unhexlify(tv_in[2].replace(' ', ''))
            tv_out[3] = tv_in[3]
            tv_out[4] = tv_in[4]
            td.append(tv_out)
        self.test_data = td

    def runTest(self):
        for data, custom, digest, text, module in self.test_data:
            hd1 = module.new(custom=custom, digest_bytes=len(digest))
            hd2 = module.new(custom=custom, digest_bytes=len(digest))
            for string in data:
                hd1.update(string)
            hd2.update(*data)
            self.assertEqual(hd1.digest(), digest, msg=text)
            self.assertEqual(hd2.digest(), digest, msg=text)