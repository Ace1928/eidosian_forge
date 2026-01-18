import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toOpenSSHECDSA(self):
    """
        L{keys.Key.toString} serializes an ECDSA key in OpenSSH format.
        """
    key = keys.Key.fromString(keydata.privateECDSA_openssh)
    self.assertEqual(key.public().toString('openssh', comment=b'comment'), keydata.publicECDSA_openssh)
    self.assertEqual(key.public().toString('openssh'), keydata.publicECDSA_openssh[:-8])