from __future__ import annotations
import errno
import math
import select as __select__
import sys
from numbers import Integral
from . import fileno
from .compat import detect_environment
class _poll:

    def __init__(self):
        self._poller = xpoll()
        self._quick_poll = self._poller.poll
        self._quick_register = self._poller.register
        self._quick_unregister = self._poller.unregister

    def register(self, fd, events):
        fd = fileno(fd)
        poll_flags = 0
        if events & ERR:
            poll_flags |= POLLERR
        if events & WRITE:
            poll_flags |= POLLOUT
        if events & READ:
            poll_flags |= POLLIN
        self._quick_register(fd, poll_flags)
        return fd

    def unregister(self, fd):
        try:
            fd = fileno(fd)
        except OSError as exc:
            if getattr(exc, 'errno', None) in SELECT_BAD_FD:
                return fd
            raise
        self._quick_unregister(fd)
        return fd

    def poll(self, timeout, round=math.ceil, POLLIN=POLLIN, POLLOUT=POLLOUT, POLLERR=POLLERR, READ=READ, WRITE=WRITE, ERR=ERR, Integral=Integral):
        timeout = 0 if timeout and timeout < 0 else round((timeout or 0) * 1000.0)
        try:
            event_list = self._quick_poll(timeout)
        except (_selecterr, OSError) as exc:
            if getattr(exc, 'errno', None) == errno.EINTR:
                return
            raise
        ready = []
        for fd, event in event_list:
            events = 0
            if event & POLLIN:
                events |= READ
            if event & POLLOUT:
                events |= WRITE
            if event & POLLERR or event & POLLNVAL or event & POLLHUP:
                events |= ERR
            assert events
            if not isinstance(fd, Integral):
                fd = fd.fileno()
            ready.append((fd, events))
        return ready

    def close(self):
        self._poller = None