import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
def test_update_after_digest(self):
    msg = b'rrrrttt'
    h = self.new()
    h.update(msg)
    dig1 = h.digest()
    self.assertRaises(TypeError, h.update, dig1)