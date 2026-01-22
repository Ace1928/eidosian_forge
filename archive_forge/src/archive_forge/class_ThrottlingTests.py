import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class ThrottlingTests(unittest.TestCase):
    """
    Tests for L{policies.ThrottlingFactory}.
    """

    def test_limit(self):
        """
        Full test using a custom server limiting number of connections.

        FIXME: https://twistedmatrix.com/trac/ticket/10012
        This is a flaky test.
        """
        server = Server()
        c1, c2, c3, c4 = (SimpleProtocol() for i in range(4))
        tServer = policies.ThrottlingFactory(server, 2)
        wrapTServer = WrappingFactory(tServer)
        wrapTServer.deferred = defer.Deferred()
        p = reactor.listenTCP(0, wrapTServer, interface='127.0.0.1')
        n = p.getHost().port

        def _connect123(results):
            reactor.connectTCP('127.0.0.1', n, SillyFactory(c1))
            c1.dConnected.addCallback(lambda r: reactor.connectTCP('127.0.0.1', n, SillyFactory(c2)))
            c2.dConnected.addCallback(lambda r: reactor.connectTCP('127.0.0.1', n, SillyFactory(c3)))
            return c3.dDisconnected

        def _check123(results):
            self.assertEqual([c.connected for c in (c1, c2, c3)], [1, 1, 1])
            self.assertEqual([c.disconnected for c in (c1, c2, c3)], [0, 0, 1])
            self.assertEqual(len(tServer.protocols.keys()), 2)
            return results

        def _lose1(results):
            c1.transport.loseConnection()
            return c1.dDisconnected

        def _connect4(results):
            reactor.connectTCP('127.0.0.1', n, SillyFactory(c4))
            return c4.dConnected

        def _check4(results):
            self.assertEqual(c4.connected, 1)
            self.assertEqual(c4.disconnected, 0)
            return results

        def _cleanup(results):
            for c in (c2, c4):
                c.transport.loseConnection()
            return defer.DeferredList([defer.maybeDeferred(p.stopListening), c2.dDisconnected, c4.dDisconnected])
        wrapTServer.deferred.addCallback(_connect123)
        wrapTServer.deferred.addCallback(_check123)
        wrapTServer.deferred.addCallback(_lose1)
        wrapTServer.deferred.addCallback(_connect4)
        wrapTServer.deferred.addCallback(_check4)
        wrapTServer.deferred.addCallback(_cleanup)
        return wrapTServer.deferred

    def test_writeSequence(self):
        """
        L{ThrottlingProtocol.writeSequence} is called on the underlying factory.
        """
        server = Server()
        tServer = TestableThrottlingFactory(task.Clock(), server)
        protocol = tServer.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 0))
        transport = StringTransportWithDisconnection()
        transport.protocol = protocol
        protocol.makeConnection(transport)
        protocol.writeSequence([b'bytes'] * 4)
        self.assertEqual(transport.value(), b'bytesbytesbytesbytes')
        self.assertEqual(tServer.writtenThisSecond, 20)

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

    def test_readLimit(self):
        """
        Check the readLimit parameter: read data and check for the pause
        status.
        """
        server = Server()
        tServer = TestableThrottlingFactory(task.Clock(), server, readLimit=10)
        port = tServer.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 0))
        tr = StringTransportWithDisconnection()
        tr.protocol = port
        port.makeConnection(tr)
        port.dataReceived(b'0123456789')
        port.dataReceived(b'abcdefghij')
        self.assertEqual(tr.value(), b'0123456789abcdefghij')
        self.assertEqual(tServer.readThisSecond, 20)
        tServer.clock.advance(1.05)
        self.assertEqual(tServer.readThisSecond, 0)
        self.assertEqual(tr.producerState, 'paused')
        tServer.clock.advance(1.05)
        self.assertEqual(tServer.readThisSecond, 0)
        self.assertEqual(tr.producerState, 'producing')
        tr.clear()
        port.dataReceived(b'0123456789')
        port.dataReceived(b'abcdefghij')
        self.assertEqual(tr.value(), b'0123456789abcdefghij')
        self.assertEqual(tServer.readThisSecond, 20)
        tServer.clock.advance(1.05)
        self.assertEqual(tServer.readThisSecond, 0)
        self.assertEqual(tr.producerState, 'paused')
        tServer.clock.advance(1.05)
        self.assertEqual(tServer.readThisSecond, 0)
        self.assertEqual(tr.producerState, 'producing')