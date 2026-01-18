import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_notEqual(self):
    """
        Test that Key objects are not-compared correctly.
        """
    rsa1 = keys.Key(self.rsaObj)
    rsa2 = keys.Key(self.rsaObj)
    rsa3 = keys.Key(keys.Key._fromRSAComponents(n=5, e=3)._keyObject)
    dsa = keys.Key(self.dsaObj)
    self.assertFalse(rsa1 != rsa2)
    self.assertTrue(rsa1 != rsa3)
    self.assertTrue(rsa1 != dsa)
    self.assertTrue(rsa1 != object)
    self.assertTrue(rsa1 != None)