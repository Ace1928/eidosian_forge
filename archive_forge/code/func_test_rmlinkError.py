from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_rmlinkError(self) -> None:
    """
        An exception raised by L{rmlink} other than C{ENOENT} is passed up
        to the caller of L{FilesystemLock.lock}.
        """

    def fakeRmlink(name: str) -> NoReturn:
        raise OSError(errno.ENOSYS, None)
    self.patch(lockfile, 'rmlink', fakeRmlink)

    def fakeKill(pid: int, signal: int) -> None:
        if signal != 0:
            raise OSError(errno.EPERM, None)
        if pid == 43125:
            raise OSError(errno.ESRCH, None)
    self.patch(lockfile, 'kill', fakeKill)
    lockf = self.mktemp()
    lockfile.symlink(str(43125), lockf)
    lock = lockfile.FilesystemLock(lockf)
    exc = self.assertRaises(OSError, lock.lock)
    self.assertEqual(exc.errno, errno.ENOSYS)
    self.assertFalse(lock.locked)