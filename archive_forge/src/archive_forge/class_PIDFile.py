from __future__ import annotations
import errno
from os import getpid, kill, name as SYSTEM_NAME
from types import TracebackType
from typing import Any, Optional, Type
from zope.interface import Interface, implementer
from twisted.logger import Logger
from twisted.python.filepath import FilePath
@implementer(IPIDFile)
class PIDFile:
    """
    Concrete implementation of L{IPIDFile}.

    This implementation is presently not supported on non-POSIX platforms.
    Specifically, calling L{PIDFile.isRunning} will raise
    L{NotImplementedError}.
    """
    _log = Logger()

    @staticmethod
    def _format(pid: int) -> bytes:
        """
        Format a PID file's content.

        @param pid: A process ID.

        @return: Formatted PID file contents.
        """
        return f'{int(pid)}\n'.encode()

    def __init__(self, filePath: FilePath[Any]) -> None:
        """
        @param filePath: The path to the PID file on disk.
        """
        self.filePath = filePath

    def read(self) -> int:
        pidString = b''
        try:
            with self.filePath.open() as fh:
                for pidString in fh:
                    break
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise NoPIDFound('PID file does not exist')
            raise
        try:
            return int(pidString)
        except ValueError:
            raise InvalidPIDFileError(f'non-integer PID value in PID file: {pidString!r}')

    def _write(self, pid: int) -> None:
        """
        Store a PID in this PID file.

        @param pid: A PID to store.

        @raise EnvironmentError: If this PID file cannot be written.
        """
        self.filePath.setContent(self._format(pid=pid))

    def writeRunningPID(self) -> None:
        self._write(getpid())

    def remove(self) -> None:
        self.filePath.remove()

    def isRunning(self) -> bool:
        try:
            pid = self.read()
        except NoPIDFound:
            return False
        if SYSTEM_NAME == 'posix':
            return self._pidIsRunningPOSIX(pid)
        else:
            raise NotImplementedError(f'isRunning is not implemented on {SYSTEM_NAME}')

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

    def __enter__(self) -> 'PIDFile':
        try:
            if self.isRunning():
                raise AlreadyRunningError()
        except StalePIDFileError:
            self._log.info('Replacing stale PID file: {log_source}')
        self.writeRunningPID()
        return self

    def __exit__(self, excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.remove()
        return None