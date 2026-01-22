import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
class HashDigestSizeSelfTest(unittest.TestCase):

    def __init__(self, hashmod, description, expected, extra_params):
        unittest.TestCase.__init__(self)
        self.hashmod = hashmod
        self.expected = expected
        self.description = description
        self.extra_params = extra_params

    def shortDescription(self):
        return self.description

    def runTest(self):
        if 'truncate' not in self.extra_params:
            self.assertTrue(hasattr(self.hashmod, 'digest_size'))
            self.assertEqual(self.hashmod.digest_size, self.expected)
        h = self.hashmod.new(**self.extra_params)
        self.assertTrue(hasattr(h, 'digest_size'))
        self.assertEqual(h.digest_size, self.expected)