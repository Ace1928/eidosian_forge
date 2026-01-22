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
def _internal_poll(self, _deadstate=None, _waitpid=_waitpid, _WNOHANG=_WNOHANG, _ECHILD=errno.ECHILD):
    """Check if child process has terminated.  Returns returncode
            attribute.

            This method is called by __del__, so it cannot reference anything
            outside of the local scope (nor can any methods it calls).

            """
    if self.returncode is None:
        if not self._waitpid_lock.acquire(False):
            return None
        try:
            if self.returncode is not None:
                return self.returncode
            pid, sts = _waitpid(self.pid, _WNOHANG)
            if pid == self.pid:
                self._handle_exitstatus(sts)
        except OSError as e:
            if _deadstate is not None:
                self.returncode = _deadstate
            elif e.errno == _ECHILD:
                self.returncode = 0
        finally:
            self._waitpid_lock.release()
    return self.returncode