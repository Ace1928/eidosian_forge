from __future__ import annotations
import contextlib
import errno
import os
import signal
import socket
from types import FrameType
from typing import Callable, Optional, Sequence
from zope.interface import Attribute, Interface, implementer
from attrs import define, frozen
from typing_extensions import Protocol, TypeAlias
from twisted.internet.interfaces import IReadDescriptor
from twisted.python import failure, log, util
from twisted.python.runtime import platformType
@implementer(IReadDescriptor)
class _FDWaker(log.Logger):
    """
    The I{self-pipe trick<http://cr.yp.to/docs/selfpipe.html>}, used to wake
    up the main loop from another thread or a signal handler.

    L{_FDWaker} is a base class for waker implementations based on
    writing to a pipe being monitored by the reactor.

    @ivar o: The file descriptor for the end of the pipe which can be
        written to wake up a reactor monitoring this waker.

    @ivar i: The file descriptor which should be monitored in order to
        be awoken by this waker.
    """
    disconnected = 0
    i: int
    o: int

    def __init__(self) -> None:
        """Initialize."""
        self.i, self.o = os.pipe()
        fdesc.setNonBlocking(self.i)
        fdesc._setCloseOnExec(self.i)
        fdesc.setNonBlocking(self.o)
        fdesc._setCloseOnExec(self.o)
        self.fileno = lambda: self.i

    def doRead(self) -> None:
        """
        Read some bytes from the pipe and discard them.
        """
        fdesc.readFromFD(self.fileno(), lambda data: None)

    def connectionLost(self, reason):
        """Close both ends of my pipe."""
        if not hasattr(self, 'o'):
            return
        for fd in (self.i, self.o):
            try:
                os.close(fd)
            except OSError:
                pass
        del self.i, self.o