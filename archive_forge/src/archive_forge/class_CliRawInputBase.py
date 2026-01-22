import io
import logging
import subprocess
import urllib.parse
from smart_open import utils
class CliRawInputBase(io.RawIOBase):
    """Reads bytes from HDFS via the "hdfs dfs" command-line interface.

    Implements the io.RawIOBase interface of the standard library.
    """

    def __init__(self, uri):
        self._uri = uri
        self._sub = subprocess.Popen(['hdfs', 'dfs', '-cat', self._uri], stdout=subprocess.PIPE)
        self.raw = None

    def close(self):
        """Flush and close this stream."""
        logger.debug('close: called')
        self._sub.terminate()
        self._sub = None

    def readable(self):
        """Return True if the stream can be read from."""
        return self._sub is not None

    def seekable(self):
        """If False, seek(), tell() and truncate() will raise IOError."""
        return False

    def detach(self):
        """Unsupported."""
        raise io.UnsupportedOperation

    def read(self, size=-1):
        """Read up to size bytes from the object and return them."""
        return self._sub.stdout.read(size)

    def read1(self, size=-1):
        """This is the same as read()."""
        return self.read(size=size)

    def readinto(self, b):
        """Read up to len(b) bytes into b, and return the number of bytes
        read."""
        data = self.read(len(b))
        if not data:
            return 0
        b[:len(data)] = data
        return len(data)