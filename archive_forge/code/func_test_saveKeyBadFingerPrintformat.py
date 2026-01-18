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
def test_saveKeyBadFingerPrintformat(self) -> None:
    """
        L{_saveKey} raises C{keys.BadFingerprintFormat} when unsupported
        formats are requested.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_rsa').path
    key = Key.fromString(privateRSA_openssh)
    with self.assertRaises(BadFingerPrintFormat) as em:
        _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'sha-base64'})
    self.assertEqual('Unsupported fingerprint format: sha-base64', em.exception.args[0])