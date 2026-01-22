from abc import ABC
from abc import abstractmethod
import errno
import os
class _FcntlInterProcessReaderWriterLockMechanism(_InterProcessReaderWriterLockMechanism):
    """Interprocess readers writer lock implementation that works on posix
    systems."""

    @staticmethod
    def trylock(lockfile, exclusive):
        if exclusive:
            flags = fcntl.LOCK_EX | fcntl.LOCK_NB
        else:
            flags = fcntl.LOCK_SH | fcntl.LOCK_NB
        try:
            fcntl.lockf(lockfile, flags)
            return True
        except (IOError, OSError) as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return False
            else:
                raise e

    @staticmethod
    def unlock(lockfile):
        fcntl.lockf(lockfile, fcntl.LOCK_UN)

    @staticmethod
    def get_handle(path):
        return open(path, 'a+')

    @staticmethod
    def close_handle(lockfile):
        lockfile.close()