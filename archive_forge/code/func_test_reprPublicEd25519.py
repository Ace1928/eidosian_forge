import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_reprPublicEd25519(self):
    """
        The repr of a L{keys.Key} contains all the OpenSSH format for an
        Ed25519 public key.
        """
    self.assertEqual(repr(keys.Key(self.ed25519Obj).public()), dedent('                <Ed25519 Public Key (256 bits)\n                attr a:\n                \tf1:16:d1:15:4a:1e:15:0e:19:5e:19:46:b5:f2:44:\n                \t0d:b2:52:a0:ae:2a:6b:23:13:73:45:fd:40:d9:57:\n                \t7b:8b>'))