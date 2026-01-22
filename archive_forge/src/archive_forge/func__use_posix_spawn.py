import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
def _use_posix_spawn():
    """Check if posix_spawn() can be used for subprocess.

    subprocess requires a posix_spawn() implementation that properly reports
    errors to the parent process, & sets errno on the following failures:

    * Process attribute actions failed.
    * File actions failed.
    * exec() failed.

    Prefer an implementation which can use vfork() in some cases for best
    performance.
    """
    if _mswindows or not hasattr(os, 'posix_spawn'):
        return False
    if sys.platform in ('darwin', 'sunos5'):
        return True
    try:
        ver = os.confstr('CS_GNU_LIBC_VERSION')
        parts = ver.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError
        libc = parts[0]
        version = tuple(map(int, parts[1].split('.')))
        if sys.platform == 'linux' and libc == 'glibc' and (version >= (2, 24)):
            return True
    except (AttributeError, ValueError, OSError):
        pass
    return False