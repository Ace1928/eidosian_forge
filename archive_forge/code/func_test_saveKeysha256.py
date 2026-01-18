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
def test_saveKeysha256(self) -> None:
    """
        L{_saveKey} will generate key fingerprint in
        L{FingerprintFormats.SHA256-BASE64} format if explicitly specified.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_rsa').path
    key = Key.fromString(privateRSA_openssh)
    _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'sha256-base64'})
    self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=SHA256_BASE64> is:\nFBTCOoknq0mHy+kpfnY9tDdcAJuWtCpuQMaV3EsvbUI=\n' % (filename, filename))
    self.assertEqual(key.fromString(base.child('id_rsa').getContent(), None, 'passphrase'), key)
    self.assertEqual(Key.fromString(base.child('id_rsa.pub').getContent()), key.public())