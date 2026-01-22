from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IOpenSSLContextFactory(Interface):
    """
    A provider of L{IOpenSSLContextFactory} is capable of generating
    L{OpenSSL.SSL.Context} classes suitable for configuring TLS on a
    connection. A provider will store enough state to be able to generate these
    contexts as needed for individual connections.

    @see: L{twisted.internet.ssl}
    """

    def getContext() -> 'OpenSSLContext':
        """
        Returns a TLS context object, suitable for securing a TLS connection.
        This context object will be appropriately customized for the connection
        based on the state in this object.

        @return: A TLS context object.
        """