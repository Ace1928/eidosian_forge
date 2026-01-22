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
class ClientEndpointTestCaseMixin:
    """
    Generic test methods to be mixed into all client endpoint test classes.
    """

    def test_interface(self):
        """
        The endpoint provides L{interfaces.IStreamClientEndpoint}
        """
        clientFactory = object()
        ep, ignoredArgs, address = self.createClientEndpoint(MemoryReactor(), clientFactory)
        self.assertTrue(verifyObject(interfaces.IStreamClientEndpoint, ep))

    def retrieveConnectedFactory(self, reactor):
        """
        Retrieve a single factory that has connected using the given reactor.
        (This behavior is valid for TCP and SSL but needs to be overridden for
        UNIX.)

        @param reactor: a L{MemoryReactor}
        """
        return self.expectedClients(reactor)[0][2]

    def test_endpointConnectSuccess(self):
        """
        A client endpoint can connect and returns a deferred who gets called
        back with a protocol instance.
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
        factory._onConnection.callback(proto)
        self.assertEqual(receivedProtos, [proto])
        expectedClients = self.expectedClients(mreactor)
        self.assertEqual(len(expectedClients), 1)
        self.assertConnectArgs(expectedClients[0], expectedArgs)

    def test_endpointConnectFailure(self):
        """
        If an endpoint tries to connect to a non-listening port it gets
        a C{ConnectError} failure.
        """
        expectedError = error.ConnectError(string='Connection Failed')
        mreactor = RaisingMemoryReactor(connectException=expectedError)
        clientFactory = object()
        ep, ignoredArgs, ignoredDest = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        receivedExceptions = []

        def checkFailure(f):
            receivedExceptions.append(f.value)
        d.addErrback(checkFailure)
        self.assertEqual(receivedExceptions, [expectedError])

    def test_endpointConnectingCancelled(self):
        """
        Calling L{Deferred.cancel} on the L{Deferred} returned from
        L{IStreamClientEndpoint.connect} is errbacked with an expected
        L{ConnectingCancelledError} exception.
        """
        mreactor = MemoryReactor()
        clientFactory = object()
        ep, ignoredArgs, address = self.createClientEndpoint(mreactor, clientFactory)
        d = ep.connect(clientFactory)
        receivedFailures = []

        def checkFailure(f):
            receivedFailures.append(f)
        d.addErrback(checkFailure)
        d.cancel()
        attemptFactory = self.retrieveConnectedFactory(mreactor)
        attemptFactory.clientConnectionFailed(None, Failure(error.UserError()))
        self.assertEqual(len(receivedFailures), 1)
        failure = receivedFailures[0]
        self.assertIsInstance(failure.value, error.ConnectingCancelledError)
        self.assertEqual(failure.value.address, address)

    def test_endpointConnectNonDefaultArgs(self):
        """
        The endpoint should pass it's connectArgs parameter to the reactor's
        listen methods.
        """
        factory = object()
        mreactor = MemoryReactor()
        ep, expectedArgs, ignoredHost = self.createClientEndpoint(mreactor, factory, **self.connectArgs())
        ep.connect(factory)
        expectedClients = self.expectedClients(mreactor)
        self.assertEqual(len(expectedClients), 1)
        self.assertConnectArgs(expectedClients[0], expectedArgs)