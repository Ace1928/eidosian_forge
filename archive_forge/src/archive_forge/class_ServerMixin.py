from zope.interface import directlyProvides
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.tls import TLSMemoryBIOFactory
class ServerMixin:
    """
    A mixin for L{twisted.internet.tcp.Server} which just marks it as a server
    for the purposes of the default TLS handshake.

    @ivar _tlsClientDefault: Always C{False}, indicating that this is a server
        connection, and by default when TLS is negotiated this class will act as
        a TLS server.
    """
    _tlsClientDefault = False