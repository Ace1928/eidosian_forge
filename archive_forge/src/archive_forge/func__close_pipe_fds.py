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
def _close_pipe_fds(self, p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite):
    devnull_fd = getattr(self, '_devnull', None)
    with contextlib.ExitStack() as stack:
        if _mswindows:
            if p2cread != -1:
                stack.callback(p2cread.Close)
            if c2pwrite != -1:
                stack.callback(c2pwrite.Close)
            if errwrite != -1:
                stack.callback(errwrite.Close)
        else:
            if p2cread != -1 and p2cwrite != -1 and (p2cread != devnull_fd):
                stack.callback(os.close, p2cread)
            if c2pwrite != -1 and c2pread != -1 and (c2pwrite != devnull_fd):
                stack.callback(os.close, c2pwrite)
            if errwrite != -1 and errread != -1 and (errwrite != devnull_fd):
                stack.callback(os.close, errwrite)
        if devnull_fd is not None:
            stack.callback(os.close, devnull_fd)
    self._closed_child_pipe_fds = True