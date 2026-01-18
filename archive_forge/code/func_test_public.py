import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_public(self):
    """
        The L{keys.Key.public} method returns a public key for both
        public and private keys.
        """
    privateRSAKey = keys.Key.fromString(keydata.privateRSA_openssh)
    publicRSAKey = keys.Key.fromString(keydata.publicRSA_openssh)
    self.assertEqual(privateRSAKey.public(), publicRSAKey.public())
    privateDSAKey = keys.Key.fromString(keydata.privateDSA_openssh)
    publicDSAKey = keys.Key.fromString(keydata.publicDSA_openssh)
    self.assertEqual(privateDSAKey.public(), publicDSAKey.public())
    privateECDSAKey = keys.Key.fromString(keydata.privateECDSA_openssh)
    publicECDSAKey = keys.Key.fromString(keydata.publicECDSA_openssh)
    self.assertEqual(privateECDSAKey.public(), publicECDSAKey.public())
    if ED25519_SUPPORTED:
        privateEd25519Key = keys.Key.fromString(keydata.privateEd25519_openssh_new)
        publicEd25519Key = keys.Key.fromString(keydata.publicEd25519_openssh)
        self.assertEqual(privateEd25519Key.public(), publicEd25519Key.public())