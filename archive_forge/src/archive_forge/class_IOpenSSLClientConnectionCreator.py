from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IOpenSSLClientConnectionCreator(Interface):
    """
    A provider of L{IOpenSSLClientConnectionCreator} can create
    L{OpenSSL.SSL.Connection} objects for TLS clients.

    @see: L{twisted.internet.ssl}

    @note: Creating OpenSSL connection objects is subtle, error-prone, and
        security-critical.  Before implementing this interface yourself,
        consider using L{twisted.internet.ssl.optionsForClientTLS} as your
        C{contextFactory}.
    """

    def clientConnectionForTLS(tlsProtocol: 'TLSMemoryBIOProtocol') -> 'OpenSSLConnection':
        """
        Create a connection for the given client protocol.

        @param tlsProtocol: the client protocol making the request.

        @return: an OpenSSL connection object configured appropriately for the
            given Twisted protocol.
        """