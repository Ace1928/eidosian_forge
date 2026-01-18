import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_writeLimit(self):
    """
        Check the writeLimit parameter: write data, and check for the pause
        status.
        """
    server = Server()
    tServer = TestableThrottlingFactory(task.Clock(), server, writeLimit=10)
    port = tServer.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 0))
    tr = StringTransportWithDisconnection()
    tr.protocol = port
    port.makeConnection(tr)
    port.producer = port.wrappedProtocol
    port.dataReceived(b'0123456789')
    port.dataReceived(b'abcdefghij')
    self.assertEqual(tr.value(), b'0123456789abcdefghij')
    self.assertEqual(tServer.writtenThisSecond, 20)
    self.assertFalse(port.wrappedProtocol.paused)
    tServer.clock.advance(1.05)
    self.assertEqual(tServer.writtenThisSecond, 0)
    self.assertTrue(port.wrappedProtocol.paused)
    tServer.clock.advance(1.05)
    self.assertEqual(tServer.writtenThisSecond, 0)
    self.assertFalse(port.wrappedProtocol.paused)