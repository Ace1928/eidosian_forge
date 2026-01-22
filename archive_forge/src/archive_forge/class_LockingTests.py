from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
class LockingTests(TestCase):

    def _symlinkErrorTest(self, errno: int) -> None:

        def fakeSymlink(source: str, dest: str) -> NoReturn:
            raise OSError(errno, None)
        self.patch(lockfile, 'symlink', fakeSymlink)
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        exc = self.assertRaises(OSError, lock.lock)
        self.assertEqual(exc.errno, errno)

    def test_symlinkError(self) -> None:
        """
        An exception raised by C{symlink} other than C{EEXIST} is passed up to
        the caller of L{FilesystemLock.lock}.
        """
        self._symlinkErrorTest(errno.ENOSYS)

    @skipIf(platform.isWindows(), 'POSIX-specific error propagation not expected on Windows.')
    def test_symlinkErrorPOSIX(self) -> None:
        """
        An L{OSError} raised by C{symlink} on a POSIX platform with an errno of
        C{EACCES} or C{EIO} is passed to the caller of L{FilesystemLock.lock}.

        On POSIX, unlike on Windows, these are unexpected errors which cannot
        be handled by L{FilesystemLock}.
        """
        self._symlinkErrorTest(errno.EACCES)
        self._symlinkErrorTest(errno.EIO)

    def test_cleanlyAcquire(self) -> None:
        """
        If the lock has never been held, it can be acquired and the C{clean}
        and C{locked} attributes are set to C{True}.
        """
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        self.assertTrue(lock.lock())
        self.assertTrue(lock.clean)
        self.assertTrue(lock.locked)

    def test_cleanlyRelease(self) -> None:
        """
        If a lock is released cleanly, it can be re-acquired and the C{clean}
        and C{locked} attributes are set to C{True}.
        """
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        self.assertTrue(lock.lock())
        lock.unlock()
        self.assertFalse(lock.locked)
        lock = lockfile.FilesystemLock(lockf)
        self.assertTrue(lock.lock())
        self.assertTrue(lock.clean)
        self.assertTrue(lock.locked)

    def test_cannotLockLocked(self) -> None:
        """
        If a lock is currently locked, it cannot be locked again.
        """
        lockf = self.mktemp()
        firstLock = lockfile.FilesystemLock(lockf)
        self.assertTrue(firstLock.lock())
        secondLock = lockfile.FilesystemLock(lockf)
        self.assertFalse(secondLock.lock())
        self.assertFalse(secondLock.locked)

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

    def test_lockReleasedBeforeCheck(self) -> None:
        """
        If the lock is initially held but then released before it can be
        examined to determine if the process which held it still exists, it is
        acquired and the C{clean} and C{locked} attributes are set to C{True}.
        """

        def fakeReadlink(name: str) -> str:
            lockfile.rmlink(lockf)
            readlinkPatch.restore()
            return lockfile.readlink(name)
        readlinkPatch = self.patch(lockfile, 'readlink', fakeReadlink)

        def fakeKill(pid: int, signal: int) -> None:
            if signal != 0:
                raise OSError(errno.EPERM, None)
            if pid == 43125:
                raise OSError(errno.ESRCH, None)
        self.patch(lockfile, 'kill', fakeKill)
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        lockfile.symlink(str(43125), lockf)
        self.assertTrue(lock.lock())
        self.assertTrue(lock.clean)
        self.assertTrue(lock.locked)

    @skipUnless(platform.isWindows(), 'special rename EIO handling only necessary and correct on Windows.')
    def test_lockReleasedDuringAcquireSymlink(self) -> None:
        """
        If the lock is released while an attempt is made to acquire
        it, the lock attempt fails and C{FilesystemLock.lock} returns
        C{False}.  This can happen on Windows when L{lockfile.symlink}
        fails with L{IOError} of C{EIO} because another process is in
        the middle of a call to L{os.rmdir} (implemented in terms of
        RemoveDirectory) which is not atomic.
        """

        def fakeSymlink(src: str, dst: str) -> NoReturn:
            raise OSError(errno.EIO, None)
        self.patch(lockfile, 'symlink', fakeSymlink)
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        self.assertFalse(lock.lock())
        self.assertFalse(lock.locked)

    @skipUnless(platform.isWindows(), 'special readlink EACCES handling only necessary and correct on Windows.')
    def test_lockReleasedDuringAcquireReadlink(self) -> None:
        """
        If the lock is initially held but is released while an attempt
        is made to acquire it, the lock attempt fails and
        L{FilesystemLock.lock} returns C{False}.
        """

        def fakeReadlink(name: str) -> NoReturn:
            raise OSError(errno.EACCES, None)
        self.patch(lockfile, 'readlink', fakeReadlink)
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        lockfile.symlink(str(43125), lockf)
        self.assertFalse(lock.lock())
        self.assertFalse(lock.locked)

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

    def test_readlinkError(self) -> None:
        """
        An exception raised by C{readlink} other than C{ENOENT} is passed up to
        the caller of L{FilesystemLock.lock}.
        """
        self._readlinkErrorTest(OSError, errno.ENOSYS)
        self._readlinkErrorTest(IOError, errno.ENOSYS)

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

    def test_lockCleanedUpConcurrently(self) -> None:
        """
        If a second process cleans up the lock after a first one checks the
        lock and finds that no process is holding it, the first process does
        not fail when it tries to clean up the lock.
        """

        def fakeRmlink(name: str) -> None:
            rmlinkPatch.restore()
            lockfile.rmlink(lockf)
            return lockfile.rmlink(name)
        rmlinkPatch = self.patch(lockfile, 'rmlink', fakeRmlink)

        def fakeKill(pid: int, signal: int) -> None:
            if signal != 0:
                raise OSError(errno.EPERM, None)
            if pid == 43125:
                raise OSError(errno.ESRCH, None)
        self.patch(lockfile, 'kill', fakeKill)
        lockf = self.mktemp()
        lock = lockfile.FilesystemLock(lockf)
        lockfile.symlink(str(43125), lockf)
        self.assertTrue(lock.lock())
        self.assertTrue(lock.clean)
        self.assertTrue(lock.locked)

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

    def test_killError(self) -> None:
        """
        If L{kill} raises an exception other than L{OSError} with errno set to
        C{ESRCH}, the exception is passed up to the caller of
        L{FilesystemLock.lock}.
        """

        def fakeKill(pid: int, signal: int) -> NoReturn:
            raise OSError(errno.EPERM, None)
        self.patch(lockfile, 'kill', fakeKill)
        lockf = self.mktemp()
        lockfile.symlink(str(43125), lockf)
        lock = lockfile.FilesystemLock(lockf)
        exc = self.assertRaises(OSError, lock.lock)
        self.assertEqual(exc.errno, errno.EPERM)
        self.assertFalse(lock.locked)

    def test_unlockOther(self) -> None:
        """
        L{FilesystemLock.unlock} raises L{ValueError} if called for a lock
        which is held by a different process.
        """
        lockf = self.mktemp()
        lockfile.symlink(str(os.getpid() + 1), lockf)
        lock = lockfile.FilesystemLock(lockf)
        self.assertRaises(ValueError, lock.unlock)

    def test_isLocked(self) -> None:
        """
        L{isLocked} returns C{True} if the named lock is currently locked,
        C{False} otherwise.
        """
        lockf = self.mktemp()
        self.assertFalse(lockfile.isLocked(lockf))
        lock = lockfile.FilesystemLock(lockf)
        self.assertTrue(lock.lock())
        self.assertTrue(lockfile.isLocked(lockf))
        lock.unlock()
        self.assertFalse(lockfile.isLocked(lockf))