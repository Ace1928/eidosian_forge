from __future__ import annotations
import errno
from os import getpid, kill, name as SYSTEM_NAME
from types import TracebackType
from typing import Any, Optional, Type
from zope.interface import Interface, implementer
from twisted.logger import Logger
from twisted.python.filepath import FilePath
@staticmethod
def _pidIsRunningPOSIX(pid: int) -> bool:
    """
        POSIX implementation for running process check.

        Determine whether there is a running process corresponding to the given
        PID.

        @param pid: The PID to check.

        @return: True if the given PID is currently running; false otherwise.

        @raise EnvironmentError: If this PID file cannot be read.
        @raise InvalidPIDFileError: If this PID file's content is invalid.
        @raise StalePIDFileError: If this PID file's content refers to a PID
            for which there is no corresponding running process.
        """
    try:
        kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            raise StalePIDFileError('PID file refers to non-existing process')
        elif e.errno == errno.EPERM:
            return True
        else:
            raise
    else:
        return True