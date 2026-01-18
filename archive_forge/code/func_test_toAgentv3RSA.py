import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toAgentv3RSA(self):
    """
        L{keys.Key.toString} serializes an RSA key in Agent v3 format.
        """
    key = keys.Key.fromString(keydata.privateRSA_openssh)
    self.assertEqual(key.toString('agentv3'), keydata.privateRSA_agentv3)