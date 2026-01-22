from __future__ import annotations
from OpenSSL import SSL
from twisted.internet import ssl
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
class ClientTLSContext(ssl.ClientContextFactory):
    """
    SSL Context Factory for client-side connections.
    """
    isClient = 1

    def getContext(self) -> SSL.Context:
        """
        Return an L{SSL.Context} to be use for client-side connections.

        Will not return a cached context.
        This is done to improve the test coverage as most implementation
        are caching the context.
        """
        return SSL.Context(SSL.SSLv23_METHOD)