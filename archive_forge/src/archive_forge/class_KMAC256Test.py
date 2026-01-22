import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KMAC128, KMAC256
class KMAC256Test(KMACTest):
    KMAC = KMAC256
    minimum_key_bits = 256
    minimum_bytes = 8
    default_bytes = 64
    oid_variant = '20'