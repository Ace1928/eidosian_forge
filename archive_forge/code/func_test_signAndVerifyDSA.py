import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_signAndVerifyDSA(self):
    """
        Signed data can be verified using DSA.
        """
    data = b'some-data'
    key = keys.Key.fromString(keydata.privateDSA_openssh)
    signature = key.sign(data)
    self.assertTrue(key.public().verify(signature, data))
    self.assertTrue(key.verify(signature, data))