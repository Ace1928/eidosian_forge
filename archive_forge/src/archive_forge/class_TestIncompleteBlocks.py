from __future__ import print_function
import unittest
from Cryptodome.Hash import SHA256
from Cryptodome.Cipher import AES
from Cryptodome.Util.py3compat import *
from binascii import hexlify
class TestIncompleteBlocks(unittest.TestCase):

    def __init__(self, use_aesni):
        unittest.TestCase.__init__(self)
        self.use_aesni = use_aesni

    def runTest(self):
        cipher = AES.new(b'4' * 16, AES.MODE_ECB, use_aesni=self.use_aesni)
        for msg_len in range(1, 16):
            self.assertRaises(ValueError, cipher.encrypt, b'1' * msg_len)
            self.assertRaises(ValueError, cipher.encrypt, b'1' * (msg_len + 16))
            self.assertRaises(ValueError, cipher.decrypt, b'1' * msg_len)
            self.assertRaises(ValueError, cipher.decrypt, b'1' * (msg_len + 16))
        self.assertEqual(cipher.encrypt(b''), b'')
        self.assertEqual(cipher.decrypt(b''), b'')