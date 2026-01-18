import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toOpenSSHRSA(self):
    """
        L{keys.Key.toString} serializes an RSA key in OpenSSH format.
        """
    key = keys.Key.fromString(keydata.privateRSA_agentv3)
    self.assertEqual(key.toString('openssh').strip(), keydata.privateRSA_openssh)
    self.assertTrue(key.toString('openssh', passphrase=b'encrypted').find(b'DEK-Info') > 0)
    self.assertEqual(key.public().toString('openssh'), keydata.publicRSA_openssh[:-8])
    self.assertEqual(key.public().toString('openssh', comment=b'comment'), keydata.publicRSA_openssh)