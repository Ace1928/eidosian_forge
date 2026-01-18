from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_unlockOther(self) -> None:
    """
        L{FilesystemLock.unlock} raises L{ValueError} if called for a lock
        which is held by a different process.
        """
    lockf = self.mktemp()
    lockfile.symlink(str(os.getpid() + 1), lockf)
    lock = lockfile.FilesystemLock(lockf)
    self.assertRaises(ValueError, lock.unlock)