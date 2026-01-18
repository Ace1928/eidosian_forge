import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_verifyDSA(self):
    """
        A known-good DSA signature verifies successfully.
        """
    key = keys.Key.fromString(keydata.publicDSA_openssh)
    self.assertTrue(key.verify(self.dsaSignature, b''))
    self.assertFalse(key.verify(self.dsaSignature, b'a'))
    self.assertFalse(key.verify(self.rsaSignature, b''))