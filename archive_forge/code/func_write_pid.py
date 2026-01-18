import atexit
import errno
import math
import numbers
import os
import platform as _platform
import signal as _signal
import sys
import warnings
from contextlib import contextmanager
from billiard.compat import close_open_fds, get_fdmax
from billiard.util import set_pdeathsig as _set_pdeathsig
from kombu.utils.compat import maybe_fileno
from kombu.utils.encoding import safe_str
from .exceptions import SecurityError, SecurityWarning, reraise
from .local import try_import
def write_pid(self):
    pid = os.getpid()
    content = f'{pid}\n'
    pidfile_fd = os.open(self.path, PIDFILE_FLAGS, PIDFILE_MODE)
    pidfile = os.fdopen(pidfile_fd, 'w')
    try:
        pidfile.write(content)
        pidfile.flush()
        try:
            os.fsync(pidfile_fd)
        except AttributeError:
            pass
    finally:
        pidfile.close()
    rfh = open(self.path)
    try:
        if rfh.read() != content:
            raise LockFailed("Inconsistency: Pidfile content doesn't match at re-read")
    finally:
        rfh.close()