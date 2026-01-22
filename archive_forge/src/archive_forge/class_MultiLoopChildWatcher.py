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
class MultiLoopChildWatcher(AbstractChildWatcher):
    """A watcher that doesn't require running loop in the main thread.

    This implementation registers a SIGCHLD signal handler on
    instantiation (which may conflict with other code that
    install own handler for this signal).

    The solution is safe but it has a significant overhead when
    handling a big number of processes (*O(n)* each time a
    SIGCHLD is received).
    """

    def __init__(self):
        self._callbacks = {}
        self._saved_sighandler = None

    def is_active(self):
        return self._saved_sighandler is not None

    def close(self):
        self._callbacks.clear()
        if self._saved_sighandler is None:
            return
        handler = signal.getsignal(signal.SIGCHLD)
        if handler != self._sig_chld:
            logger.warning('SIGCHLD handler was changed by outside code')
        else:
            signal.signal(signal.SIGCHLD, self._saved_sighandler)
        self._saved_sighandler = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_child_handler(self, pid, callback, *args):
        loop = events.get_running_loop()
        self._callbacks[pid] = (loop, callback, args)
        self._do_waitpid(pid)

    def remove_child_handler(self, pid):
        try:
            del self._callbacks[pid]
            return True
        except KeyError:
            return False

    def attach_loop(self, loop):
        if self._saved_sighandler is not None:
            return
        self._saved_sighandler = signal.signal(signal.SIGCHLD, self._sig_chld)
        if self._saved_sighandler is None:
            logger.warning('Previous SIGCHLD handler was set by non-Python code, restore to default handler on watcher close.')
            self._saved_sighandler = signal.SIG_DFL
        signal.siginterrupt(signal.SIGCHLD, False)

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
            debug_log = False
        else:
            if pid == 0:
                return
            returncode = waitstatus_to_exitcode(status)
            debug_log = True
        try:
            loop, callback, args = self._callbacks.pop(pid)
        except KeyError:
            logger.warning('Child watcher got an unexpected pid: %r', pid, exc_info=True)
        else:
            if loop.is_closed():
                logger.warning('Loop %r that handles pid %r is closed', loop, pid)
            else:
                if debug_log and loop.get_debug():
                    logger.debug('process %s exited with returncode %s', expected_pid, returncode)
                loop.call_soon_threadsafe(callback, pid, returncode, *args)

    def _sig_chld(self, signum, frame):
        try:
            self._do_waitpid_all()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException:
            logger.warning('Unknown exception in SIGCHLD handler', exc_info=True)