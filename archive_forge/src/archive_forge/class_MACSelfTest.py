import re
import sys
import unittest
import binascii
import Cryptodome.Hash
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes
from Cryptodome.Util.strxor import strxor_c
class MACSelfTest(unittest.TestCase):

    def __init__(self, module, description, result, data, key, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.result = t2b(result)
        self.data = t2b(data)
        self.key = t2b(key)
        self.params = params
        self.description = description

    def shortDescription(self):
        return self.description

    def runTest(self):
        result_hex = hexlify(self.result)
        h = self.module.new(self.key, **self.params)
        h.update(self.data)
        self.assertEqual(self.result, h.digest())
        self.assertEqual(hexlify(self.result).decode('ascii'), h.hexdigest())
        h.verify(self.result)
        h.hexverify(result_hex)
        wrong_mac = strxor_c(self.result, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)
        self.assertRaises(ValueError, h.hexverify, '4556')
        h = self.module.new(self.key, self.data, **self.params)
        self.assertEqual(self.result, h.digest())
        self.assertEqual(hexlify(self.result).decode('ascii'), h.hexdigest())
        try:
            h = self.module.new(self.key, self.data, **self.params)
            h2 = h.copy()
            h3 = h.copy()
            h2.update(b'bla')
            self.assertEqual(h3.digest(), self.result)
            h.update(b'bla')
            self.assertEqual(h.digest(), h2.digest())
        except NotImplementedError:
            pass
        self.assertTrue(isinstance(h.digest(), type(b'')))
        self.assertTrue(isinstance(h.hexdigest(), type('')))
        h.hexverify(h.hexdigest())
        h.hexverify(h.hexdigest().encode('ascii'))