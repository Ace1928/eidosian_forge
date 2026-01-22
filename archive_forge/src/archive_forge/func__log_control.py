import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
def _log_control(self, s):
    """Write control characters to the appropriate log files"""
    if self.encoding is not None:
        s = s.decode(self.encoding, 'replace')
    self._log(s, 'send')