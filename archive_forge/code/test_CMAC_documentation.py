import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Hash import CMAC
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
Verify that internal caching is implemented correctly