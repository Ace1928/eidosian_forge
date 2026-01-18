from __future__ import division
import collections
import contextlib
import datetime
import functools
import os
import signal
import subprocess
import sys
import threading
import time
from . import _common
from ._common import AIX
from ._common import BSD
from ._common import CONN_CLOSE
from ._common import CONN_CLOSE_WAIT
from ._common import CONN_CLOSING
from ._common import CONN_ESTABLISHED
from ._common import CONN_FIN_WAIT1
from ._common import CONN_FIN_WAIT2
from ._common import CONN_LAST_ACK
from ._common import CONN_LISTEN
from ._common import CONN_NONE
from ._common import CONN_SYN_RECV
from ._common import CONN_SYN_SENT
from ._common import CONN_TIME_WAIT
from ._common import FREEBSD  # NOQA
from ._common import LINUX
from ._common import MACOS
from ._common import NETBSD  # NOQA
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import OPENBSD  # NOQA
from ._common import OSX  # deprecated alias
from ._common import POSIX  # NOQA
from ._common import POWER_TIME_UNKNOWN
from ._common import POWER_TIME_UNLIMITED
from ._common import STATUS_DEAD
from ._common import STATUS_DISK_SLEEP
from ._common import STATUS_IDLE
from ._common import STATUS_LOCKED
from ._common import STATUS_PARKED
from ._common import STATUS_RUNNING
from ._common import STATUS_SLEEPING
from ._common import STATUS_STOPPED
from ._common import STATUS_TRACING_STOP
from ._common import STATUS_WAITING
from ._common import STATUS_WAKING
from ._common import STATUS_ZOMBIE
from ._common import SUNOS
from ._common import WINDOWS
from ._common import AccessDenied
from ._common import Error
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import ZombieProcess
from ._common import memoize_when_activated
from ._common import wrap_numbers as _wrap_numbers
from ._compat import PY3 as _PY3
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import SubprocessTimeoutExpired as _SubprocessTimeoutExpired
from ._compat import long
def virtual_memory():
    """Return statistics about system memory usage as a namedtuple
    including the following fields, expressed in bytes:

     - total:
       total physical memory available.

     - available:
       the memory that can be given instantly to processes without the
       system going into swap.
       This is calculated by summing different memory values depending
       on the platform and it is supposed to be used to monitor actual
       memory usage in a cross platform fashion.

     - percent:
       the percentage usage calculated as (total - available) / total * 100

     - used:
        memory used, calculated differently depending on the platform and
        designed for informational purposes only:
        macOS: active + wired
        BSD: active + wired + cached
        Linux: total - free

     - free:
       memory not being used at all (zeroed) that is readily available;
       note that this doesn't reflect the actual memory available
       (use 'available' instead)

    Platform-specific fields:

     - active (UNIX):
       memory currently in use or very recently used, and so it is in RAM.

     - inactive (UNIX):
       memory that is marked as not used.

     - buffers (BSD, Linux):
       cache for things like file system metadata.

     - cached (BSD, macOS):
       cache for various things.

     - wired (macOS, BSD):
       memory that is marked to always stay in RAM. It is never moved to disk.

     - shared (BSD):
       memory that may be simultaneously accessed by multiple processes.

    The sum of 'used' and 'available' does not necessarily equal total.
    On Windows 'available' and 'free' are the same.
    """
    global _TOTAL_PHYMEM
    ret = _psplatform.virtual_memory()
    _TOTAL_PHYMEM = ret.total
    return ret