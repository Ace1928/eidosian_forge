import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toLSHRSA(self):
    """
        L{keys.Key.toString} serializes an RSA key in LSH format.
        """
    key = keys.Key.fromString(keydata.privateRSA_openssh)
    self.assertEqual(key.toString('lsh'), keydata.privateRSA_lsh)
    self.assertEqual(key.public().toString('lsh'), keydata.publicRSA_lsh)