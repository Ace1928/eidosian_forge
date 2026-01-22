from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
class HostnameEndpointsOneIPv4Tests(ClientEndpointTestCaseMixin, unittest.TestCase):
    """
    Tests for the hostname based endpoints when GAI returns only one
    (IPv4) address.
    """

    def createClientEndpoint(self, reactor, clientFactory, **connectArgs):
        """
        Creates a L{HostnameEndpoint} instance where the hostname is resolved
        into a single IPv4 address.
        """
        expectedAddress = '1.2.3.4'
        address = HostnameAddress(b'example.com', 80)
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(reactor, [expectedAddress]), b'example.com', address.port, **connectArgs)
        return (endpoint, (expectedAddress, address.port, clientFactory, connectArgs.get('timeout', 30), connectArgs.get('bindAddress', None)), address)

    def expectedClients(self, reactor):
        """
        @return: List of calls to L{IReactorTCP.connectTCP}
        """
        return reactor.tcpClients

    def assertConnectArgs(self, receivedArgs, expectedArgs):
        """
        Compare host, port, timeout, and bindAddress in C{receivedArgs}
        to C{expectedArgs}.  We ignore the factory because we don't
        only care what protocol comes out of the
        C{IStreamClientEndpoint.connect} call.

        @param receivedArgs: C{tuple} of (C{host}, C{port}, C{factory},
            C{timeout}, C{bindAddress}) that was passed to
            L{IReactorTCP.connectTCP}.
        @param expectedArgs: C{tuple} of (C{host}, C{port}, C{factory},
            C{timeout}, C{bindAddress}) that we expect to have been passed
            to L{IReactorTCP.connectTCP}.
        """
        host, port, ignoredFactory, timeout, bindAddress = receivedArgs
        expectedHost, expectedPort, _ignoredFactory, expectedTimeout, expectedBindAddress = expectedArgs
        self.assertEqual(host, expectedHost)
        self.assertEqual(port, expectedPort)
        self.assertEqual(timeout, expectedTimeout)
        self.assertEqual(bindAddress, expectedBindAddress)

    def connectArgs(self):
        """
        @return: C{dict} of keyword arguments to pass to connect.
        """
        return {'timeout': 10, 'bindAddress': ('localhost', 49595)}

    def test_endpointConnectingCancelled(self, advance=None):
        """
        Calling L{Deferred.cancel} on the L{Deferred} returned from
        L{IStreamClientEndpoint.connect} will cause it to be errbacked with a
        L{ConnectingCancelledError} exception.
        """
        mreactor = MemoryReactor()
        clientFactory = protocol.Factory()
        clientFactory.protocol = protocol.Protocol
        ep, ignoredArgs, address = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        if advance is not None:
            mreactor.advance(advance)
        d.cancel()
        attemptFactory = self.retrieveConnectedFactory(mreactor)
        attemptFactory.clientConnectionFailed(None, Failure(error.UserError()))
        failure = self.failureResultOf(d)
        self.assertIsInstance(failure.value, error.ConnectingCancelledError)
        self.assertEqual(failure.value.address, address)
        self.assertTrue(mreactor.tcpClients[0][2]._connector.stoppedConnecting)
        self.assertEqual([], mreactor.getDelayedCalls())

    def test_endpointConnectingCancelledAfterAllAttemptsStarted(self):
        """
        Calling L{Deferred.cancel} on the L{Deferred} returned from
        L{IStreamClientEndpoint.connect} after enough time has passed that all
        connection attempts have been initiated will cause it to be errbacked
        with a L{ConnectingCancelledError} exception.
        """
        oneBetween = endpoints.HostnameEndpoint._DEFAULT_ATTEMPT_DELAY
        advance = oneBetween + oneBetween / 2.0
        self.test_endpointConnectingCancelled(advance=advance)

    def test_endpointConnectFailure(self):
        """
        If L{HostnameEndpoint.connect} is invoked and there is no server
        listening for connections, the returned L{Deferred} will fail with
        C{ConnectError}.
        """
        expectedError = error.ConnectError(string='Connection Failed')
        mreactor = RaisingMemoryReactorWithClock(connectException=expectedError)
        clientFactory = object()
        ep, ignoredArgs, ignoredDest = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        mreactor.advance(endpoints.HostnameEndpoint._DEFAULT_ATTEMPT_DELAY)
        self.assertEqual(self.failureResultOf(d).value, expectedError)
        self.assertEqual([], mreactor.getDelayedCalls())

    def test_endpointConnectFailureAfterIteration(self):
        """
        If a connection attempt initiated by
        L{HostnameEndpoint.connect} fails only after
        L{HostnameEndpoint} has exhausted the list of possible server
        addresses, the returned L{Deferred} will fail with
        C{ConnectError}.
        """
        expectedError = error.ConnectError(string='Connection Failed')
        mreactor = MemoryReactor()
        clientFactory = object()
        ep, ignoredArgs, ignoredDest = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        mreactor.advance(0.3)
        host, port, factory, timeout, bindAddress = mreactor.tcpClients[0]
        factory.clientConnectionFailed(mreactor.connectors[0], expectedError)
        self.assertEqual(self.failureResultOf(d).value, expectedError)
        self.assertEqual([], mreactor.getDelayedCalls())

    def test_endpointConnectSuccessAfterIteration(self):
        """
        If a connection attempt initiated by
        L{HostnameEndpoint.connect} succeeds only after
        L{HostnameEndpoint} has exhausted the list of possible server
        addresses, the returned L{Deferred} will fire with the
        connected protocol instance and the endpoint will leave no
        delayed calls in the reactor.
        """
        proto = object()
        mreactor = MemoryReactor()
        clientFactory = object()
        ep, expectedArgs, ignoredDest = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        receivedProtos = []

        def checkProto(p):
            receivedProtos.append(p)
        d.addCallback(checkProto)
        factory = self.retrieveConnectedFactory(mreactor)
        mreactor.advance(0.3)
        factory._onConnection.callback(proto)
        self.assertEqual(receivedProtos, [proto])
        expectedClients = self.expectedClients(mreactor)
        self.assertEqual(len(expectedClients), 1)
        self.assertConnectArgs(expectedClients[0], expectedArgs)
        self.assertEqual([], mreactor.getDelayedCalls())