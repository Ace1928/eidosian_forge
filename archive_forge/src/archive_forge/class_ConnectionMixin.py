from zope.interface import directlyProvides
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.tls import TLSMemoryBIOFactory
class ConnectionMixin:
    """
    A mixin for L{twisted.internet.abstract.FileDescriptor} which adds an
    L{ITLSTransport} implementation.

    @ivar TLS: A flag indicating whether TLS is currently in use on this
        transport.  This is not a good way for applications to check for TLS,
        instead use L{twisted.internet.interfaces.ISSLTransport}.
    """
    TLS = False

    def startTLS(self, ctx, normal=True):
        """
        @see: L{ITLSTransport.startTLS}
        """
        startTLS(self, ctx, normal, FileDescriptor)

    def write(self, bytes):
        """
        Write some bytes to this connection, passing them through a TLS layer if
        necessary, or discarding them if the connection has already been lost.
        """
        if self.TLS:
            if self.connected:
                self.protocol.write(bytes)
        else:
            FileDescriptor.write(self, bytes)

    def writeSequence(self, iovec):
        """
        Write some bytes to this connection, scatter/gather-style, passing them
        through a TLS layer if necessary, or discarding them if the connection
        has already been lost.
        """
        if self.TLS:
            if self.connected:
                self.protocol.writeSequence(iovec)
        else:
            FileDescriptor.writeSequence(self, iovec)

    def loseConnection(self):
        """
        Close this connection after writing all pending data.

        If TLS has been negotiated, perform a TLS shutdown.
        """
        if self.TLS:
            if self.connected and (not self.disconnecting):
                self.protocol.loseConnection()
        else:
            FileDescriptor.loseConnection(self)

    def registerProducer(self, producer, streaming):
        """
        Register a producer.

        If TLS is enabled, the TLS connection handles this.
        """
        if self.TLS:
            self.protocol.registerProducer(producer, streaming)
        else:
            FileDescriptor.registerProducer(self, producer, streaming)

    def unregisterProducer(self):
        """
        Unregister a producer.

        If TLS is enabled, the TLS connection handles this.
        """
        if self.TLS:
            self.protocol.unregisterProducer()
        else:
            FileDescriptor.unregisterProducer(self)