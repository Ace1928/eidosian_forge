from __future__ import annotations
import os
import sys
from contextlib import suppress
from errno import ENOSYS
from typing import cast
from ._api import BaseFileLock
from ._util import ensure_directory_exists
def _acquire(self) -> None:
    ensure_directory_exists(self.lock_file)
    open_flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    fd = os.open(self.lock_file, open_flags, self._context.mode)
    with suppress(PermissionError):
        os.fchmod(fd, self._context.mode)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exception:
        os.close(fd)
        if exception.errno == ENOSYS:
            msg = 'FileSystem does not appear to support flock; user SoftFileLock instead'
            raise NotImplementedError(msg) from exception
    else:
        self._context.lock_file_fd = fd