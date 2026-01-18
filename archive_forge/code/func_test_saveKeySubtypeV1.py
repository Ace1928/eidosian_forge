from __future__ import annotations
import getpass
import os
import subprocess
import sys
from io import StringIO
from typing import Callable
from typing_extensions import NoReturn
from twisted.conch.test.keydata import (
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_saveKeySubtypeV1(self) -> None:
    """
        L{_saveKey} can be told to write the new private key file in OpenSSH
        v1 format.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_rsa').path
    key = Key.fromString(privateRSA_openssh)
    _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex', 'private-key-subtype': 'v1'})
    self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\n85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da\n' % (filename, filename))
    privateKeyContent = base.child('id_rsa').getContent()
    self.assertEqual(key.fromString(privateKeyContent, None, 'passphrase'), key)
    self.assertTrue(privateKeyContent.startswith(b'-----BEGIN OPENSSH PRIVATE KEY-----\n'))
    self.assertEqual(Key.fromString(base.child('id_rsa.pub').getContent()), key.public())