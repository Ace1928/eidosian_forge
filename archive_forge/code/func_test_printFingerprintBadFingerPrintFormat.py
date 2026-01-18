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
def test_printFingerprintBadFingerPrintFormat(self) -> None:
    """
        L{printFigerprint} raises C{keys.BadFingerprintFormat} when unsupported
        formats are requested.
        """
    filename = self.mktemp()
    FilePath(filename).setContent(publicRSA_openssh)
    with self.assertRaises(BadFingerPrintFormat) as em:
        printFingerprint({'filename': filename, 'format': 'sha-base64'})
    self.assertEqual('Unsupported fingerprint format: sha-base64', em.exception.args[0])