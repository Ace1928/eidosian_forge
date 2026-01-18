from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipUnless(platform.isWindows(), 'special readlink EACCES handling only necessary and correct on Windows.')
def test_readlinkEACCESWindows(self) -> None:
    """
        L{lockfile.readlink} raises L{OSError} with C{errno} set to L{EACCES}
        on Windows when the underlying file open attempt fails with C{EACCES}.

        Opening a file on Windows may fail if the path is inside a directory
        which is in the process of being deleted (directory deletion appears
        not to be atomic).
        """
    name = self.mktemp()

    def fakeOpen(path: str, mode: str) -> NoReturn:
        raise OSError(errno.EACCES, None)
    self.patch(lockfile, '_open', fakeOpen)
    exc = self.assertRaises(IOError, lockfile.readlink, name)
    self.assertEqual(exc.errno, errno.EACCES)