from __future__ import annotations
from zope.interface import implementedBy, implementer, implementer_only
from OpenSSL import SSL
from twisted.internet import interfaces, tcp
from twisted.internet._sslverify import (
@implementer(interfaces.IOpenSSLContextFactory)
class ClientContextFactory:
    """A context factory for SSL clients."""
    isClient = 1
    method = SSL.TLS_METHOD
    _contextFactory = SSL.Context

    def getContext(self):
        ctx = self._contextFactory(self.method)
        ctx.set_options(SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1)
        return ctx