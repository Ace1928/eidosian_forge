import contextlib
import errno
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from . import debug, errors, osutils, trace
from .hooks import Hooks
from .i18n import gettext
from .transport import Transport
class _fcntl_TemporaryWriteLock(_OSLock):
    """A token used when grabbing a temporary_write_lock.

        Call restore_read_lock() when you are done with the write lock.
        """

    def __init__(self, read_lock):
        super().__init__()
        self._read_lock = read_lock
        self.filename = read_lock.filename
        count = _fcntl_ReadLock._open_locks[self.filename]
        if count > 1:
            raise errors.LockContention(self.filename)
        if self.filename in _fcntl_WriteLock._open_locks:
            raise AssertionError('file already locked: %r' % (self.filename,))
        try:
            new_f = open(self.filename, 'rb+')
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EPERM):
                raise errors.LockFailed(self.filename, str(e))
            raise
        try:
            fcntl.lockf(new_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise errors.LockContention(self.filename, e)
        _fcntl_WriteLock._open_locks.add(self.filename)
        self.f = new_f

    def restore_read_lock(self):
        """Restore the original ReadLock."""
        fcntl.lockf(self.f, fcntl.LOCK_UN)
        self._clear_f()
        _fcntl_WriteLock._open_locks.remove(self.filename)
        read_lock = self._read_lock
        self._read_lock = None
        return read_lock