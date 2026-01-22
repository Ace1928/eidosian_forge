import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class TCPConnectorTestsBuilder(ReactorBuilder):
    """
    Tests for the L{IConnector} provider returned by L{IReactorTCP.connectTCP}.
    """
    requiredInterfaces = (IReactorTCP,)

    def test_connectorIdentity(self):
        """
        L{IReactorTCP.connectTCP} returns an object which provides
        L{IConnector}.  The destination of the connector is the address which
        was passed to C{connectTCP}.  The same connector object is passed to
        the factory's C{startedConnecting} method as to the factory's
        C{clientConnectionLost} method.
        """
        serverFactory = ClosingFactory()
        reactor = self.buildReactor()
        tcpPort = reactor.listenTCP(0, serverFactory, interface=self.interface)
        serverFactory.port = tcpPort
        portNumber = tcpPort.getHost().port
        seenConnectors = []
        seenFailures = []
        clientFactory = ClientStartStopFactory()
        clientFactory.clientConnectionLost = lambda connector, reason: (seenConnectors.append(connector), seenFailures.append(reason))
        clientFactory.startedConnecting = seenConnectors.append
        connector = reactor.connectTCP(self.interface, portNumber, clientFactory)
        self.assertTrue(IConnector.providedBy(connector))
        dest = connector.getDestination()
        self.assertEqual(dest.type, 'TCP')
        self.assertEqual(dest.host, self.interface)
        self.assertEqual(dest.port, portNumber)
        clientFactory.whenStopped.addBoth(lambda _: reactor.stop())
        self.runReactor(reactor)
        seenFailures[0].trap(ConnectionDone)
        self.assertEqual(seenConnectors, [connector, connector])

    def test_userFail(self):
        """
        Calling L{IConnector.stopConnecting} in C{Factory.startedConnecting}
        results in C{Factory.clientConnectionFailed} being called with
        L{error.UserError} as the reason.
        """
        serverFactory = MyServerFactory()
        reactor = self.buildReactor()
        tcpPort = reactor.listenTCP(0, serverFactory, interface=self.interface)
        portNumber = tcpPort.getHost().port
        fatalErrors = []

        def startedConnecting(connector):
            try:
                connector.stopConnecting()
            except Exception:
                fatalErrors.append(Failure())
                reactor.stop()
        clientFactory = ClientStartStopFactory()
        clientFactory.startedConnecting = startedConnecting
        clientFactory.whenStopped.addBoth(lambda _: reactor.stop())
        reactor.callWhenRunning(lambda: reactor.connectTCP(self.interface, portNumber, clientFactory))
        self.runReactor(reactor)
        if fatalErrors:
            self.fail(fatalErrors[0].getTraceback())
        clientFactory.reason.trap(UserError)
        self.assertEqual(clientFactory.failed, 1)

    def test_reconnect(self):
        """
        Calling L{IConnector.connect} in C{Factory.clientConnectionLost} causes
        a new connection attempt to be made.
        """
        serverFactory = ClosingFactory()
        reactor = self.buildReactor()
        tcpPort = reactor.listenTCP(0, serverFactory, interface=self.interface)
        serverFactory.port = tcpPort
        portNumber = tcpPort.getHost().port
        clientFactory = MyClientFactory()

        def clientConnectionLost(connector, reason):
            connector.connect()
        clientFactory.clientConnectionLost = clientConnectionLost
        reactor.connectTCP(self.interface, portNumber, clientFactory)
        protocolMadeAndClosed = []

        def reconnectFailed(ignored):
            p = clientFactory.protocol
            protocolMadeAndClosed.append((p.made, p.closed))
            reactor.stop()
        clientFactory.failDeferred.addCallback(reconnectFailed)
        self.runReactor(reactor)
        clientFactory.reason.trap(ConnectionRefusedError)
        self.assertEqual(protocolMadeAndClosed, [(1, 1)])