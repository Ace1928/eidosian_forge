import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_verifyDSANoPrefix(self):
    """
        Some commercial SSH servers send DSA keys as 2 20-byte numbers;
        they are still verified as valid keys.
        """
    key = keys.Key.fromString(keydata.publicDSA_openssh)
    self.assertTrue(key.verify(self.dsaSignature[-40:], b''))