from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class ITLSTransport(ITCPTransport):
    """
    A TCP transport that supports switching to TLS midstream.

    Once TLS mode is started the transport will implement L{ISSLTransport}.
    """

    def startTLS(contextFactory: Union[IOpenSSLClientConnectionCreator, IOpenSSLServerConnectionCreator]) -> None:
        """
        Initiate TLS negotiation.

        @param contextFactory: An object which creates appropriately configured
            TLS connections.

            For clients, use L{twisted.internet.ssl.optionsForClientTLS}; for
            servers, use L{twisted.internet.ssl.CertificateOptions}.

        @type contextFactory: L{IOpenSSLClientConnectionCreator} or
            L{IOpenSSLServerConnectionCreator}, depending on whether this
            L{ITLSTransport} is a server or not.  If the appropriate interface
            is not provided by the value given for C{contextFactory}, it must
            be an implementor of L{IOpenSSLContextFactory}.
        """