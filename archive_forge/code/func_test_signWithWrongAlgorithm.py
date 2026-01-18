import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_signWithWrongAlgorithm(self):
    """
        L{keys.Key.sign} raises L{keys.BadSignatureAlgorithmError} when
        asked to sign with a public key algorithm that doesn't make sense
        with the given key.
        """
    key = keys.Key.fromString(keydata.privateRSA_openssh)
    self.assertRaises(keys.BadSignatureAlgorithmError, key.sign, b'some data', signatureType=b'ssh-dss')
    key = keys.Key.fromString(keydata.privateECDSA_openssh)
    self.assertRaises(keys.BadSignatureAlgorithmError, key.sign, b'some data', signatureType=b'ssh-dss')