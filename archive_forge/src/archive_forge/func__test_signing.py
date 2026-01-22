import os
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _test_signing(self, dsaObj):
    k = bytes_to_long(a2b_hex(self.k))
    m_hash = bytes_to_long(a2b_hex(self.m_hash))
    r = bytes_to_long(a2b_hex(self.r))
    s = bytes_to_long(a2b_hex(self.s))
    r_out, s_out = dsaObj._sign(m_hash, k)
    self.assertEqual((r, s), (r_out, s_out))