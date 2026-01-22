from __future__ import print_function
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util.strxor import strxor
class NISTTestVectorsGCM(unittest.TestCase):

    def __init__(self, a):
        self.use_clmul = True
        unittest.TestCase.__init__(self, a)