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
def _posix_spawn(self, args, executable, env, restore_signals, p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite):
    """Execute program using os.posix_spawn()."""
    if env is None:
        env = os.environ
    kwargs = {}
    if restore_signals:
        sigset = []
        for signame in ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ'):
            signum = getattr(signal, signame, None)
            if signum is not None:
                sigset.append(signum)
        kwargs['setsigdef'] = sigset
    file_actions = []
    for fd in (p2cwrite, c2pread, errread):
        if fd != -1:
            file_actions.append((os.POSIX_SPAWN_CLOSE, fd))
    for fd, fd2 in ((p2cread, 0), (c2pwrite, 1), (errwrite, 2)):
        if fd != -1:
            file_actions.append((os.POSIX_SPAWN_DUP2, fd, fd2))
    if file_actions:
        kwargs['file_actions'] = file_actions
    self.pid = os.posix_spawn(executable, args, env, **kwargs)
    self._child_created = True
    self._close_pipe_fds(p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite)