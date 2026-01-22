import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import BLAKE2b, BLAKE2s
class Blake2bOfficialTestVector(Blake2OfficialTestVector):
    BLAKE2 = BLAKE2b
    name = 'BLAKE2b'
    max_bytes = 64