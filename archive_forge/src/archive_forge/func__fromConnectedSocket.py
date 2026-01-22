from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
@classmethod
def _fromConnectedSocket(cls, fileDescriptor, addressFamily, factory, reactor):
    """
        Create a new L{Server} based on an existing connected I{SOCK_STREAM}
        socket.

        Arguments are the same as to L{Server.__init__}, except where noted.

        @param fileDescriptor: An integer file descriptor associated with a
            connected socket.  The socket must be in non-blocking mode.  Any
            additional attributes desired, such as I{FD_CLOEXEC}, must also be
            set already.

        @param addressFamily: The address family (sometimes called I{domain})
            of the existing socket.  For example, L{socket.AF_INET}.

        @return: A new instance of C{cls} wrapping the socket given by
            C{fileDescriptor}.
        """
    addressType = address.IPv4Address
    if addressFamily == socket.AF_INET6:
        addressType = address.IPv6Address
    skt = socket.fromfd(fileDescriptor, addressFamily, socket.SOCK_STREAM)
    addr = _getpeername(skt)
    protocolAddr = addressType('TCP', *addr)
    localPort = skt.getsockname()[1]
    protocol = factory.buildProtocol(protocolAddr)
    if protocol is None:
        skt.close()
        return
    self = cls(skt, protocol, addr, None, addr[1], reactor)
    self.repstr = '<{} #{} on {}>'.format(self.protocol.__class__.__name__, self.sessionno, localPort)
    protocol.makeConnection(self)
    return self