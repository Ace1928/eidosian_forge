import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_signAndVerifyEC(self):
    """
        Signed data can be verified using EC.
        """
    data = b'some-data'
    key = keys.Key.fromString(keydata.privateECDSA_openssh)
    signature = key.sign(data)
    key384 = keys.Key.fromString(keydata.privateECDSA_openssh384)
    signature384 = key384.sign(data)
    key521 = keys.Key.fromString(keydata.privateECDSA_openssh521)
    signature521 = key521.sign(data)
    self.assertTrue(key.public().verify(signature, data))
    self.assertTrue(key.verify(signature, data))
    self.assertTrue(key384.public().verify(signature384, data))
    self.assertTrue(key384.verify(signature384, data))
    self.assertTrue(key521.public().verify(signature521, data))
    self.assertTrue(key521.verify(signature521, data))