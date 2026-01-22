from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IHandshakeListener(Interface):
    """
    An interface implemented by a L{IProtocol} to indicate that it would like
    to be notified when TLS handshakes complete when run over a TLS-based
    transport.

    This interface is only guaranteed to be called when run over a TLS-based
    transport: non TLS-based transports will not respect this interface.
    """

    def handshakeCompleted() -> None:
        """
        Notification of the TLS handshake being completed.

        This notification fires when OpenSSL has completed the TLS handshake.
        At this point the TLS connection is established, and the protocol can
        interrogate its transport (usually an L{ISSLTransport}) for details of
        the TLS connection.

        This notification *also* fires whenever the TLS session is
        renegotiated. As a result, protocols that have certain minimum security
        requirements should implement this interface to ensure that they are
        able to re-evaluate the security of the TLS session if it changes.
        """