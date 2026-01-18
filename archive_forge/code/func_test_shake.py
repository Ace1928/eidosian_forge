import unittest
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import cSHAKE128, cSHAKE256, SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, tobytes
def test_shake(self):
    for digest_len in range(64):
        xof1 = self.cshake.new(b'TEST')
        xof2 = self.shake.new(b'TEST')
        self.assertEqual(xof1.read(digest_len), xof2.read(digest_len))