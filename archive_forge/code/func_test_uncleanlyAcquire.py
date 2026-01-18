from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_uncleanlyAcquire(self) -> None:
    """
        If a lock was held by a process which no longer exists, it can be
        acquired, the C{clean} attribute is set to C{False}, and the
        C{locked} attribute is set to C{True}.
        """
    owner = 12345

    def fakeKill(pid: int, signal: int) -> None:
        if signal != 0:
            raise OSError(errno.EPERM, None)
        if pid == owner:
            raise OSError(errno.ESRCH, None)
    lockf = self.mktemp()
    self.patch(lockfile, 'kill', fakeKill)
    lockfile.symlink(str(owner), lockf)
    lock = lockfile.FilesystemLock(lockf)
    self.assertTrue(lock.lock())
    self.assertFalse(lock.clean)
    self.assertTrue(lock.locked)
    self.assertEqual(lockfile.readlink(lockf), str(os.getpid()))