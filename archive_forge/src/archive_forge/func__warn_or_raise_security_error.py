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
def _warn_or_raise_security_error(egid, euid, gid, uid, pickle_or_serialize):
    c_force_root = os.environ.get('C_FORCE_ROOT', False)
    if pickle_or_serialize and (not c_force_root):
        raise SecurityError(ROOT_DISALLOWED.format(uid=uid, euid=euid, gid=gid, egid=egid))
    warnings.warn(SecurityWarning(ROOT_DISCOURAGED.format(uid=uid, euid=euid, gid=gid, egid=egid)))