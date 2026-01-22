from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
class StringUDPSocket:
    """
    A fake UDP socket object, which returns a fixed sequence of strings and/or
    socket errors.  Useful for testing.

    @ivar retvals: A C{list} containing either strings or C{socket.error}s.

    @ivar connectedAddr: The address the socket is connected to.
    """

    def __init__(self, retvals: list[bytes | socket.error]) -> None:
        self.retvals = retvals
        self.connectedAddr: object | None = None

    def connect(self, addr: object) -> None:
        self.connectedAddr = addr

    def recvfrom(self, size: int) -> tuple[bytes, None]:
        """
        Return (or raise) the next value from C{self.retvals}.
        """
        ret = self.retvals.pop(0)
        if isinstance(ret, socket.error):
            raise ret
        return (ret, None)