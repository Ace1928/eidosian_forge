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
def test_saveKeyEmptyPassphrase(self) -> None:
    """
        L{_saveKey} will choose an empty string for the passphrase if
        no-passphrase is C{True}.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_rsa').path
    key = Key.fromString(privateRSA_openssh)
    _saveKey(key, {'filename': filename, 'no-passphrase': True, 'format': 'md5-hex'})
    self.assertEqual(key.fromString(base.child('id_rsa').getContent(), None, b''), key)