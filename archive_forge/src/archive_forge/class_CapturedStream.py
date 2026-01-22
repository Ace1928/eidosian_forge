import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
class CapturedStream(object):
    """A temporarily redirected output stream."""

    def __init__(self, stream, filename):
        self._stream = stream
        self._fd = stream.fileno()
        self._filename = filename
        self._uncaptured_fd = os.dup(self._fd)
        cap_fd = os.open(self._filename, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 384)
        self._stream.flush()
        os.dup2(cap_fd, self._fd)
        os.close(cap_fd)

    def RestartCapture(self):
        """Resume capturing output to a file (after calling StopCapture)."""
        assert self._uncaptured_fd
        cap_fd = os.open(self._filename, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 384)
        self._stream.flush()
        os.dup2(cap_fd, self._fd)
        os.close(cap_fd)

    def StopCapture(self):
        """Remove output redirection."""
        self._stream.flush()
        os.dup2(self._uncaptured_fd, self._fd)

    def filename(self):
        return self._filename

    def __del__(self):
        self.StopCapture()
        os.close(self._uncaptured_fd)
        del self._uncaptured_fd