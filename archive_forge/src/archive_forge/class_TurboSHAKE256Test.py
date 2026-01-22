import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
class TurboSHAKE256Test(TurboSHAKETest):
    TurboSHAKE = TurboSHAKE256