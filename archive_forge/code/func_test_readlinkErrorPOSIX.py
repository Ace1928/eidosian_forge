from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(platform.isWindows(), 'POSIX-specific error propagation not expected on Windows.')
def test_readlinkErrorPOSIX(self) -> None:
    """
        Any L{IOError} raised by C{readlink} on a POSIX platform passed to the
        caller of L{FilesystemLock.lock}.

        On POSIX, unlike on Windows, these are unexpected errors which cannot
        be handled by L{FilesystemLock}.
        """
    self._readlinkErrorTest(IOError, errno.ENOSYS)
    self._readlinkErrorTest(IOError, errno.EACCES)