import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def _check_encryption(self, rsaObj):
    plaintext = a2b_hex(self.plaintext)
    ciphertext = a2b_hex(self.ciphertext)
    new_ciphertext2 = rsaObj._encrypt(bytes_to_long(plaintext))
    self.assertEqual(bytes_to_long(ciphertext), new_ciphertext2)