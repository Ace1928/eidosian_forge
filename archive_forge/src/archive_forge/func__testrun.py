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
def _testrun(self, keyType: str, keySize: str | None=None, privateKeySubtype: str | None=None) -> None:
    filename = self.mktemp()
    args = ['ckeygen', '-t', keyType, '-f', filename, '--no-passphrase']
    if keySize is not None:
        args.extend(['-b', keySize])
    if privateKeySubtype is not None:
        args.extend(['--private-key-subtype', privateKeySubtype])
    subprocess.call(args)
    privKey = Key.fromFile(filename)
    pubKey = Key.fromFile(filename + '.pub')
    if keyType == 'ecdsa':
        self.assertEqual(privKey.type(), 'EC')
    elif keyType == 'ed25519':
        self.assertEqual(privKey.type(), 'Ed25519')
    else:
        self.assertEqual(privKey.type(), keyType.upper())
    self.assertTrue(pubKey.isPublic())