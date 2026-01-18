import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
Test vectors from https://gitlab.com/dkg/ocb-test-vectors