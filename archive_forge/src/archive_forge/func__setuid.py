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
def _setuid(uid, gid):
    if not gid and pwd:
        gid = pwd.getpwuid(uid).pw_gid
    setgid(gid)
    initgroups(uid, gid)
    setuid(uid)
    try:
        setuid(0)
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise
    else:
        raise SecurityError('non-root user able to restore privileges after setuid.')