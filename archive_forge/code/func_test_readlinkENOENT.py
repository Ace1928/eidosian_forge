from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_readlinkENOENT(self) -> None:
    """
        L{lockfile.readlink} raises L{OSError} with C{errno} set to L{ENOENT}
        when an attempt is made to read a symlink which does not exist.
        """
    name = self.mktemp()
    exc = self.assertRaises(OSError, lockfile.readlink, name)
    self.assertEqual(exc.errno, errno.ENOENT)