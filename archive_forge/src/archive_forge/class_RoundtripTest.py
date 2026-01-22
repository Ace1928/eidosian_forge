import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
class RoundtripTest(unittest.TestCase):

    def __init__(self, module, params):
        from Cryptodome import Random
        unittest.TestCase.__init__(self)
        self.module = module
        self.iv = Random.get_random_bytes(module.block_size)
        self.key = b(params['key'])
        self.plaintext = 100 * b(params['plaintext'])
        self.module_name = params.get('module_name', None)

    def shortDescription(self):
        return '%s .decrypt() output of .encrypt() should not be garbled' % (self.module_name,)

    def runTest(self):
        mode = self.module.MODE_ECB
        encryption_cipher = self.module.new(a2b_hex(self.key), mode)
        ciphertext = encryption_cipher.encrypt(self.plaintext)
        decryption_cipher = self.module.new(a2b_hex(self.key), mode)
        decrypted_plaintext = decryption_cipher.decrypt(ciphertext)
        self.assertEqual(self.plaintext, decrypted_plaintext)