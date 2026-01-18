import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_privateBlobDSA(self):
    """
        L{keys.Key.privateBlob} returns the SSH protocol-level format of a DSA
        private key.
        """
    publicNumbers = self.dsaObj.private_numbers().public_numbers
    self.assertEqual(keys.Key(self.dsaObj).privateBlob(), common.NS(b'ssh-dss') + common.MP(publicNumbers.parameter_numbers.p) + common.MP(publicNumbers.parameter_numbers.q) + common.MP(publicNumbers.parameter_numbers.g) + common.MP(publicNumbers.y) + common.MP(self.dsaObj.private_numbers().x))