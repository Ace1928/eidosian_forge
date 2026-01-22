from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
def _createConnection(self, tlsProtocol):
    """
        Create an OpenSSL connection and set it up good.

        @param tlsProtocol: The protocol which is establishing the connection.
        @type tlsProtocol: L{TLSMemoryBIOProtocol}

        @return: an OpenSSL connection object for C{tlsProtocol} to use
        @rtype: L{OpenSSL.SSL.Connection}
        """
    connectionCreator = self._connectionCreator
    if self._creatorInterface is IOpenSSLClientConnectionCreator:
        connection = connectionCreator.clientConnectionForTLS(tlsProtocol)
        self._applyProtocolNegotiation(connection)
        connection.set_connect_state()
    else:
        connection = connectionCreator.serverConnectionForTLS(tlsProtocol)
        self._applyProtocolNegotiation(connection)
        connection.set_accept_state()
    return connection