import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_transportInterfaces(self):
    """
        The transport wrapper passed to the wrapped protocol's
        C{makeConnection} provides the same interfaces as are provided by the
        original transport.
        """

    class IStubTransport(Interface):
        pass

    @implementer(IStubTransport)
    class StubTransport:
        pass
    implementedBy(policies.ProtocolWrapper)
    proto = protocol.Protocol()
    wrapper = policies.ProtocolWrapper(policies.WrappingFactory(None), proto)
    wrapper.makeConnection(StubTransport())
    self.assertTrue(IStubTransport.providedBy(proto.transport))