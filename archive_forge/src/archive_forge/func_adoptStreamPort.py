from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def adoptStreamPort(fileDescriptor: int, addressFamily: 'AddressFamily', factory: 'ServerFactory') -> 'IListeningPort':
    """
        Add an existing listening I{SOCK_STREAM} socket to the reactor to
        monitor for new connections to accept and handle.

        @param fileDescriptor: A file descriptor associated with a socket which
            is already bound to an address and marked as listening.  The socket
            must be set non-blocking.  Any additional flags (for example,
            close-on-exec) must also be set by application code.  Application
            code is responsible for closing the file descriptor, which may be
            done as soon as C{adoptStreamPort} returns.
        @param addressFamily: The address family (or I{domain}) of the socket.
            For example, L{socket.AF_INET6}.
        @param factory: A L{ServerFactory} instance to use to create new
            protocols to handle connections accepted via this socket.

        @return: An object providing L{IListeningPort}.

        @raise twisted.internet.error.UnsupportedAddressFamily: If the
            given address family is not supported by this reactor, or
            not supported with the given socket type.
        @raise twisted.internet.error.UnsupportedSocketType: If the
            given socket type is not supported by this reactor, or not
            supported with the given socket type.
        """