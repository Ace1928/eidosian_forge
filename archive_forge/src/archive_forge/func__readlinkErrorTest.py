from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def _readlinkErrorTest(self, exceptionType: type[OSError] | type[IOError], errno: int) -> None:

    def fakeReadlink(name: str) -> NoReturn:
        raise exceptionType(errno, None)
    self.patch(lockfile, 'readlink', fakeReadlink)
    lockf = self.mktemp()
    lockfile.symlink(str(43125), lockf)
    lock = lockfile.FilesystemLock(lockf)
    exc = self.assertRaises(exceptionType, lock.lock)
    self.assertEqual(exc.errno, errno)
    self.assertFalse(lock.locked)