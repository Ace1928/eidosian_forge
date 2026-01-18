import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toStringNormalizesUnicodePassphrase(self):
    """
        L{keys.Key.toString} applies Normalization Form KC to Unicode
        passphrases.
        """
    key = keys.Key(self.rsaObj)
    key_data = key.toString('openssh', passphrase='verschlüsselt')
    self.assertEqual(keys.Key.fromString(key_data, passphrase='verschlüsselt'.encode()), key)
    self.assertRaises(keys.PassphraseNormalizationError, key.toString, 'openssh', passphrase='unassigned \uffff')