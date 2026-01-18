import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toLSHDSA(self):
    """
        L{keys.Key.toString} serializes a DSA key in LSH format.
        """
    key = keys.Key.fromString(keydata.privateDSA_openssh)
    self.assertEqual(key.toString('lsh'), keydata.privateDSA_lsh)
    self.assertEqual(key.public().toString('lsh'), keydata.publicDSA_lsh)