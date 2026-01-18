from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_symlinkEEXIST(self) -> None:
    """
        L{lockfile.symlink} raises L{OSError} with C{errno} set to L{EEXIST}
        when an attempt is made to create a symlink which already exists.
        """
    name = self.mktemp()
    lockfile.symlink('foo', name)
    exc = self.assertRaises(OSError, lockfile.symlink, 'foo', name)
    self.assertEqual(exc.errno, errno.EEXIST)