import binascii
import unittest
from Cryptodome.Util import RFC1751
from Cryptodome.Util.py3compat import *
class RFC1751Test_e2k(unittest.TestCase):

    def runTest(self):
        """Check converting English strings to keys"""
        for key, words in test_data:
            key = binascii.a2b_hex(b(key))
            self.assertEqual(RFC1751.english_to_key(words), key)