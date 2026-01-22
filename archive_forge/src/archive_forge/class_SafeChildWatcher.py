import errno
import io
import itertools
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import selector_events
from . import tasks
from . import transports
from .log import logger
class SafeChildWatcher(BaseChildWatcher):
    """'Safe' child watcher implementation.

    This implementation avoids disrupting other code spawning processes by
    polling explicitly each process in the SIGCHLD handler instead of calling
    os.waitpid(-1).

    This is a safe solution but it has a significant overhead when handling a
    big number of children (O(n) each time SIGCHLD is raised)
    """

    def close(self):
        self._callbacks.clear()
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        pass

    def add_child_handler(self, pid, callback, *args):
        self._callbacks[pid] = (callback, args)
        self._do_waitpid(pid)

    def remove_child_handler(self, pid):
        try:
            del self._callbacks[pid]
            return True
        except KeyError:
            return False

    def _do_waitpid_all(self):
        for pid in list(self._callbacks):
            self._do_waitpid(pid)

    def _do_waitpid(self, expected_pid):
        assert expected_pid > 0
        try:
            pid, status = os.waitpid(expected_pid, os.WNOHANG)
        except ChildProcessError:
            pid = expected_pid
            returncode = 255
            logger.warning('Unknown child process pid %d, will report returncode 255', pid)
        else:
            if pid == 0:
                return
            returncode = waitstatus_to_exitcode(status)
            if self._loop.get_debug():
                logger.debug('process %s exited with returncode %s', expected_pid, returncode)
        try:
            callback, args = self._callbacks.pop(pid)
        except KeyError:
            if self._loop.get_debug():
                logger.warning('Child watcher got an unexpected pid: %r', pid, exc_info=True)
        else:
            callback(pid, returncode, *args)