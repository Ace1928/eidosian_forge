from __future__ import annotations
from zope.interface import implementedBy, implementer, implementer_only
from OpenSSL import SSL
from twisted.internet import interfaces, tcp
from twisted.internet._sslverify import (
class DefaultOpenSSLContextFactory(ContextFactory):
    """
    L{DefaultOpenSSLContextFactory} is a factory for server-side SSL context
    objects.  These objects define certain parameters related to SSL
    handshakes and the subsequent connection.

    @ivar _contextFactory: A callable which will be used to create new
        context objects.  This is typically L{OpenSSL.SSL.Context}.
    """
    _context = None

    def __init__(self, privateKeyFileName, certificateFileName, sslmethod=SSL.TLS_METHOD, _contextFactory=SSL.Context):
        """
        @param privateKeyFileName: Name of a file containing a private key
        @param certificateFileName: Name of a file containing a certificate
        @param sslmethod: The SSL method to use
        """
        self.privateKeyFileName = privateKeyFileName
        self.certificateFileName = certificateFileName
        self.sslmethod = sslmethod
        self._contextFactory = _contextFactory
        self.cacheContext()

    def cacheContext(self):
        if self._context is None:
            ctx = self._contextFactory(self.sslmethod)
            ctx.set_options(SSL.OP_NO_SSLv2)
            ctx.use_certificate_file(self.certificateFileName)
            ctx.use_privatekey_file(self.privateKeyFileName)
            self._context = ctx

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_context']
        return d

    def __setstate__(self, state):
        self.__dict__ = state

    def getContext(self):
        """
        Return an SSL context.
        """
        return self._context