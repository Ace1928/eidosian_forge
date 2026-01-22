import unittest
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.Cipher import ARC2
class BufferOverflowTest(unittest.TestCase):

    def runTest(self):
        """ARC2 with keylength > 128"""
        key = b('x') * 16384
        self.assertRaises(ValueError, ARC2.new, key, ARC2.MODE_ECB)