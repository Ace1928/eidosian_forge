import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_overriddenCallLater(self):
    """
        Test that the callLater of the clock is used instead of
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        """
    self.proto.setTimeout(10)
    self.assertEqual(len(self.clock.calls), 1)