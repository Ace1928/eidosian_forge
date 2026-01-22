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
def RestartCapture(self):
    """Resume capturing output to a file (after calling StopCapture)."""
    assert self._uncaptured_fd
    cap_fd = os.open(self._filename, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 384)
    self._stream.flush()
    os.dup2(cap_fd, self._fd)
    os.close(cap_fd)