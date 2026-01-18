import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_protocolLogPrefixFallback(self):
    """
        If the wrapped protocol doesn't have a L{logPrefix} method,
        L{ProtocolWrapper.logPrefix} falls back to the protocol class name.
        """

    class NoProtocol:
        pass
    server = Server()
    server.protocol = NoProtocol
    factory = policies.WrappingFactory(server)
    protocol = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 35))
    self.assertEqual('NoProtocol (ProtocolWrapper)', protocol.logPrefix())