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
@implementer(_IWaker)
class _SocketWaker(log.Logger):
    """
    The I{self-pipe trick<http://cr.yp.to/docs/selfpipe.html>}, implemented
    using a pair of sockets rather than pipes (due to the lack of support in
    select() on Windows for pipes), used to wake up the main loop from
    another thread.
    """
    disconnected = 0

    def __init__(self) -> None:
        """Initialize."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as server:
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            client.connect(server.getsockname())
            reader, clientaddr = server.accept()
        client.setblocking(False)
        reader.setblocking(False)
        self.r = reader
        self.w = client
        self.fileno = self.r.fileno

    def wakeUp(self):
        """Send a byte to my connection."""
        try:
            util.untilConcludes(self.w.send, b'x')
        except OSError as e:
            if e.args[0] != errno.WSAEWOULDBLOCK:
                raise

    def doRead(self):
        """
        Read some data from my connection.
        """
        try:
            self.r.recv(8192)
        except OSError:
            pass

    def connectionLost(self, reason):
        self.r.close()
        self.w.close()