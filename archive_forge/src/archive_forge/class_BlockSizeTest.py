import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
class BlockSizeTest(unittest.TestCase):

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.key = a2b_hex(b(params['key']))

    def runTest(self):
        cipher = self.module.new(self.key, self.module.MODE_ECB)
        self.assertEqual(cipher.block_size, self.module.block_size)