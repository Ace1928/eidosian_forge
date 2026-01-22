import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
class BaseLogFile:
    """
    The base class for a log file that can be rotated.
    """
    synchronized = ['write', 'rotate']

    def __init__(self, name: str, directory: str, defaultMode: Optional[int]=None) -> None:
        """
        Create a log file.

        @param name: name of the file
        @param directory: directory holding the file
        @param defaultMode: permissions used to create the file. Default to
        current permissions of the file if the file exists.
        """
        self.directory = directory
        self.name = name
        self.path = os.path.join(directory, name)
        if defaultMode is None and os.path.exists(self.path):
            self.defaultMode: Optional[int] = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
        else:
            self.defaultMode = defaultMode
        self._openFile()

    @classmethod
    def fromFullPath(cls, filename, *args, **kwargs):
        """
        Construct a log file from a full file path.
        """
        logPath = os.path.abspath(filename)
        return cls(os.path.basename(logPath), os.path.dirname(logPath), *args, **kwargs)

    def shouldRotate(self):
        """
        Override with a method to that returns true if the log
        should be rotated.
        """
        raise NotImplementedError

    def _openFile(self):
        """
        Open the log file.

        The log file is always opened in binary mode.
        """
        self.closed = False
        if os.path.exists(self.path):
            self._file = cast(BinaryIO, open(self.path, 'rb+', 0))
            self._file.seek(0, 2)
        elif self.defaultMode is not None:
            oldUmask = os.umask(511)
            try:
                self._file = cast(BinaryIO, open(self.path, 'wb+', 0))
            finally:
                os.umask(oldUmask)
        else:
            self._file = cast(BinaryIO, open(self.path, 'wb+', 0))
        if self.defaultMode is not None:
            try:
                os.chmod(self.path, self.defaultMode)
            except OSError:
                pass

    def write(self, data):
        """
        Write some data to the file.

        @param data: The data to write.  Text will be encoded as UTF-8.
        @type data: L{bytes} or L{unicode}
        """
        if self.shouldRotate():
            self.flush()
            self.rotate()
        if isinstance(data, str):
            data = data.encode('utf8')
        self._file.write(data)

    def flush(self):
        """
        Flush the file.
        """
        self._file.flush()

    def close(self):
        """
        Close the file.

        The file cannot be used once it has been closed.
        """
        self.closed = True
        self._file.close()
        del self._file

    def reopen(self):
        """
        Reopen the log file. This is mainly useful if you use an external log
        rotation tool, which moves under your feet.

        Note that on Windows you probably need a specific API to rename the
        file, as it's not supported to simply use os.rename, for example.
        """
        self.close()
        self._openFile()

    def getCurrentLog(self):
        """
        Return a LogReader for the current log file.
        """
        return LogReader(self.path)