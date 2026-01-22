import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
class ListeningTests(TestCase):

    def test_listen(self):
        """
        L{IReactorTCP.listenTCP} returns an object which provides
        L{IListeningPort}.
        """
        f = MyServerFactory()
        p1 = reactor.listenTCP(0, f, interface='127.0.0.1')
        self.addCleanup(p1.stopListening)
        self.assertTrue(interfaces.IListeningPort.providedBy(p1))

    def testStopListening(self):
        """
        The L{IListeningPort} returned by L{IReactorTCP.listenTCP} can be
        stopped with its C{stopListening} method.  After the L{Deferred} it
        (optionally) returns has been called back, the port number can be bound
        to a new server.
        """
        f = MyServerFactory()
        port = reactor.listenTCP(0, f, interface='127.0.0.1')
        n = port.getHost().port

        def cbStopListening(ignored):
            port = reactor.listenTCP(n, f, interface='127.0.0.1')
            return port.stopListening()
        d = defer.maybeDeferred(port.stopListening)
        d.addCallback(cbStopListening)
        return d

    def testNumberedInterface(self):
        f = MyServerFactory()
        p1 = reactor.listenTCP(0, f, interface='127.0.0.1')
        return p1.stopListening()

    def testPortRepr(self):
        f = MyServerFactory()
        p = reactor.listenTCP(0, f)
        portNo = str(p.getHost().port)
        self.assertFalse(repr(p).find(portNo) == -1)

        def stoppedListening(ign):
            self.assertFalse(repr(p).find(portNo) != -1)
        d = defer.maybeDeferred(p.stopListening)
        return d.addCallback(stoppedListening)

    def test_serverRepr(self):
        """
        Check that the repr string of the server transport get the good port
        number if the server listens on 0.
        """
        server = MyServerFactory()
        serverConnMade = server.protocolConnectionMade = defer.Deferred()
        port = reactor.listenTCP(0, server)
        self.addCleanup(port.stopListening)
        client = MyClientFactory()
        clientConnMade = client.protocolConnectionMade = defer.Deferred()
        connector = reactor.connectTCP('127.0.0.1', port.getHost().port, client)
        self.addCleanup(connector.disconnect)

        def check(result):
            serverProto, clientProto = result
            portNumber = port.getHost().port
            self.assertEqual(repr(serverProto.transport), f'<AccumulatingProtocol #0 on {portNumber}>')
            serverProto.transport.loseConnection()
            clientProto.transport.loseConnection()
        return defer.gatherResults([serverConnMade, clientConnMade]).addCallback(check)

    def test_restartListening(self):
        """
        Stop and then try to restart a L{tcp.Port}: after a restart, the
        server should be able to handle client connections.
        """
        serverFactory = MyServerFactory()
        port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
        self.addCleanup(port.stopListening)

        def cbStopListening(ignored):
            port.startListening()
            client = MyClientFactory()
            serverFactory.protocolConnectionMade = defer.Deferred()
            client.protocolConnectionMade = defer.Deferred()
            connector = reactor.connectTCP('127.0.0.1', port.getHost().port, client)
            self.addCleanup(connector.disconnect)
            return defer.gatherResults([serverFactory.protocolConnectionMade, client.protocolConnectionMade]).addCallback(close)

        def close(result):
            serverProto, clientProto = result
            clientProto.transport.loseConnection()
            serverProto.transport.loseConnection()
        d = defer.maybeDeferred(port.stopListening)
        d.addCallback(cbStopListening)
        return d

    def test_exceptInStop(self):
        """
        If the server factory raises an exception in C{stopFactory}, the
        deferred returned by L{tcp.Port.stopListening} should fail with the
        corresponding error.
        """
        serverFactory = MyServerFactory()

        def raiseException():
            raise RuntimeError('An error')
        serverFactory.stopFactory = raiseException
        port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
        return self.assertFailure(port.stopListening(), RuntimeError)

    def test_restartAfterExcept(self):
        """
        Even if the server factory raise an exception in C{stopFactory}, the
        corresponding C{tcp.Port} instance should be in a sane state and can
        be restarted.
        """
        serverFactory = MyServerFactory()

        def raiseException():
            raise RuntimeError('An error')
        serverFactory.stopFactory = raiseException
        port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
        self.addCleanup(port.stopListening)

        def cbStopListening(ignored):
            del serverFactory.stopFactory
            port.startListening()
            client = MyClientFactory()
            serverFactory.protocolConnectionMade = defer.Deferred()
            client.protocolConnectionMade = defer.Deferred()
            connector = reactor.connectTCP('127.0.0.1', port.getHost().port, client)
            self.addCleanup(connector.disconnect)
            return defer.gatherResults([serverFactory.protocolConnectionMade, client.protocolConnectionMade]).addCallback(close)

        def close(result):
            serverProto, clientProto = result
            clientProto.transport.loseConnection()
            serverProto.transport.loseConnection()
        return self.assertFailure(port.stopListening(), RuntimeError).addCallback(cbStopListening)

    def test_directConnectionLostCall(self):
        """
        If C{connectionLost} is called directly on a port object, it succeeds
        (and doesn't expect the presence of a C{deferred} attribute).

        C{connectionLost} is called by L{reactor.disconnectAll} at shutdown.
        """
        serverFactory = MyServerFactory()
        port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
        portNumber = port.getHost().port
        port.connectionLost(None)
        client = MyClientFactory()
        serverFactory.protocolConnectionMade = defer.Deferred()
        client.protocolConnectionMade = defer.Deferred()
        reactor.connectTCP('127.0.0.1', portNumber, client)

        def check(ign):
            client.reason.trap(error.ConnectionRefusedError)
        return client.failDeferred.addCallback(check)

    def test_exceptInConnectionLostCall(self):
        """
        If C{connectionLost} is called directory on a port object and that the
        server factory raises an exception in C{stopFactory}, the exception is
        passed through to the caller.

        C{connectionLost} is called by L{reactor.disconnectAll} at shutdown.
        """
        serverFactory = MyServerFactory()

        def raiseException():
            raise RuntimeError('An error')
        serverFactory.stopFactory = raiseException
        port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
        self.assertRaises(RuntimeError, port.connectionLost, None)