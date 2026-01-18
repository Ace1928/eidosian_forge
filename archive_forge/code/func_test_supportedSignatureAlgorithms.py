import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_supportedSignatureAlgorithms(self):
    """
        L{keys.Key.supportedSignatureAlgorithms} returns the appropriate
        public key signature algorithms for each key type.
        """
    self.assertEqual(keys.Key(self.rsaObj).supportedSignatureAlgorithms(), [b'rsa-sha2-512', b'rsa-sha2-256', b'ssh-rsa'])
    self.assertEqual(keys.Key(self.dsaObj).supportedSignatureAlgorithms(), [b'ssh-dss'])
    self.assertEqual(keys.Key(self.ecObj).supportedSignatureAlgorithms(), [b'ecdsa-sha2-nistp256'])
    if ED25519_SUPPORTED:
        self.assertEqual(keys.Key(self.ed25519Obj).supportedSignatureAlgorithms(), [b'ssh-ed25519'])
    self.assertRaises(RuntimeError, keys.Key(None).supportedSignatureAlgorithms)
    self.assertRaises(RuntimeError, keys.Key(self).supportedSignatureAlgorithms)