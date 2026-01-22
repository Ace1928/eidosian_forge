from __future__ import absolute_import
import errno
import os
import time
from . import (LockBase, AlreadyLocked, LockFailed, NotLocked, NotMyLock,
class PIDLockFile(LockBase):
    """ Lockfile implemented as a Unix PID file.

    The lock file is a normal file named by the attribute `path`.
    A lock's PID file contains a single line of text, containing
    the process ID (PID) of the process that acquired the lock.

    >>> lock = PIDLockFile('somefile')
    >>> lock = PIDLockFile('somefile')
    """

    def __init__(self, path, threaded=False, timeout=None):
        LockBase.__init__(self, path, False, timeout)
        self.unique_name = self.path

    def read_pid(self):
        """ Get the PID from the lock file.
            """
        return read_pid_from_pidfile(self.path)

    def is_locked(self):
        """ Test if the lock is currently held.

            The lock is held if the PID file for this lock exists.

            """
        return os.path.exists(self.path)

    def i_am_locking(self):
        """ Test if the lock is held by the current process.

        Returns ``True`` if the current process ID matches the
        number stored in the PID file.
        """
        return self.is_locked() and os.getpid() == self.read_pid()

    def acquire(self, timeout=None):
        """ Acquire the lock.

        Creates the PID file for this lock, or raises an error if
        the lock could not be acquired.
        """
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout
        while True:
            try:
                write_pid_to_pidfile(self.path)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    if time.time() > end_time:
                        if timeout is not None and timeout > 0:
                            raise LockTimeout('Timeout waiting to acquire lock for %s' % self.path)
                        else:
                            raise AlreadyLocked('%s is already locked' % self.path)
                    time.sleep(timeout is not None and timeout / 10 or 0.1)
                else:
                    raise LockFailed('failed to create %s' % self.path)
            else:
                return

    def release(self):
        """ Release the lock.

            Removes the PID file to release the lock, or raises an
            error if the current process does not hold the lock.

            """
        if not self.is_locked():
            raise NotLocked('%s is not locked' % self.path)
        if not self.i_am_locking():
            raise NotMyLock('%s is locked, but not by me' % self.path)
        remove_existing_pidfile(self.path)

    def break_lock(self):
        """ Break an existing lock.

            Removes the PID file if it already exists, otherwise does
            nothing.

            """
        remove_existing_pidfile(self.path)