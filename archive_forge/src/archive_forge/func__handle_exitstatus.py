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
def _handle_exitstatus(self, sts, _waitstatus_to_exitcode=_waitstatus_to_exitcode, _WIFSTOPPED=_WIFSTOPPED, _WSTOPSIG=_WSTOPSIG):
    """All callers to this function MUST hold self._waitpid_lock."""
    if _WIFSTOPPED(sts):
        self.returncode = -_WSTOPSIG(sts)
    else:
        self.returncode = _waitstatus_to_exitcode(sts)