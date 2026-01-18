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
def test_saveKey(self) -> None:
    """
        L{_saveKey} writes the private and public parts of a key to two
        different files and writes a report of this to standard out.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    filename = base.child('id_rsa').path
    key = Key.fromString(privateRSA_openssh)
    _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex'})
    self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\n85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da\n' % (filename, filename))
    self.assertEqual(key.fromString(base.child('id_rsa').getContent(), None, 'passphrase'), key)
    self.assertEqual(Key.fromString(base.child('id_rsa.pub').getContent()), key.public())