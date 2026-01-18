import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import ChaCha20_Poly1305
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
Test cases from RFC7539