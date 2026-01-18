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
def test_saveKeyEd25519(self) -> None:
    """
        L{_saveKey} writes the private and public parts of a key to two
        different files and writes a report of this to standard out.
        Test with Ed25519 key.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_ed25519').path
    key = Key.fromString(privateEd25519_openssh_new)
    _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex'})
    self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\nab:ee:c8:ed:e5:01:1b:45:b7:8d:b2:f0:8f:61:1c:14\n' % (filename, filename))
    self.assertEqual(key.fromString(base.child('id_ed25519').getContent(), None, 'passphrase'), key)
    self.assertEqual(Key.fromString(base.child('id_ed25519.pub').getContent()), key.public())