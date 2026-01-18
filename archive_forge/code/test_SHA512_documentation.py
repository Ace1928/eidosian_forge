from binascii import hexlify
from Cryptodome.Hash import SHA512
from .common import make_hash_tests
from Cryptodome.SelfTest.loader import load_test_vectors
Self-test suite for Cryptodome.Hash.SHA512