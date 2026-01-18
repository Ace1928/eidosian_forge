import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_reprPublicECDSA(self):
    """
        The repr of a L{keys.Key} contains all the OpenSSH format for an ECDSA
        public key.
        """
    self.assertEqual(repr(keys.Key(self.ecObj).public()), dedent('                <Elliptic Curve Public Key (256 bits)\n                curve:\n                \tecdsa-sha2-nistp256\n                x:\n                \t{x}\n                y:\n                \t{y}>\n                ').format(**keydata.ECDatanistp256))