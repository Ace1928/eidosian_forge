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
class UNIXEndpointsTests(EndpointTestCaseMixin, unittest.TestCase):
    """
    Tests for UnixSocket Endpoints.
    """

    def retrieveConnectedFactory(self, reactor):
        """
        Override L{EndpointTestCaseMixin.retrieveConnectedFactory} to account
        for different index of 'factory' in C{connectUNIX} args.
        """
        return self.expectedClients(reactor)[0][1]

    def expectedServers(self, reactor):
        """
        @return: List of calls to L{IReactorUNIX.listenUNIX}
        """
        return reactor.unixServers

    def expectedClients(self, reactor):
        """
        @return: List of calls to L{IReactorUNIX.connectUNIX}
        """
        return reactor.unixClients

    def assertConnectArgs(self, receivedArgs, expectedArgs):
        """
        Compare path, timeout, checkPID in C{receivedArgs} to C{expectedArgs}.
        We ignore the factory because we don't only care what protocol comes
        out of the C{IStreamClientEndpoint.connect} call.

        @param receivedArgs: C{tuple} of (C{path}, C{timeout}, C{checkPID})
            that was passed to L{IReactorUNIX.connectUNIX}.
        @param expectedArgs: C{tuple} of (C{path}, C{timeout}, C{checkPID})
            that we expect to have been passed to L{IReactorUNIX.connectUNIX}.
        """
        path, ignoredFactory, timeout, checkPID = receivedArgs
        expectedPath, _ignoredFactory, expectedTimeout, expectedCheckPID = expectedArgs
        self.assertEqual(path, expectedPath)
        self.assertEqual(timeout, expectedTimeout)
        self.assertEqual(checkPID, expectedCheckPID)

    def connectArgs(self):
        """
        @return: C{dict} of keyword arguments to pass to connect.
        """
        return {'timeout': 10, 'checkPID': 1}

    def listenArgs(self):
        """
        @return: C{dict} of keyword arguments to pass to listen
        """
        return {'backlog': 100, 'mode': 384, 'wantPID': 1}

    def createServerEndpoint(self, reactor, factory, **listenArgs):
        """
        Create an L{UNIXServerEndpoint} and return the tools to verify its
        behaviour.

        @param reactor: A fake L{IReactorUNIX} that L{UNIXServerEndpoint} can
            call L{IReactorUNIX.listenUNIX} on.
        @param factory: The thing that we expect to be passed to our
            L{IStreamServerEndpoint.listen} implementation.
        @param listenArgs: Optional dictionary of arguments to
            L{IReactorUNIX.listenUNIX}.
        """
        address = UNIXAddress(self.mktemp())
        return (endpoints.UNIXServerEndpoint(reactor, address.name, **listenArgs), (address.name, factory, listenArgs.get('backlog', 50), listenArgs.get('mode', 438), listenArgs.get('wantPID', 0)), address)

    def createClientEndpoint(self, reactor, clientFactory, **connectArgs):
        """
        Create an L{UNIXClientEndpoint} and return the values needed to verify
        its behaviour.

        @param reactor: A fake L{IReactorUNIX} that L{UNIXClientEndpoint} can
            call L{IReactorUNIX.connectUNIX} on.
        @param clientFactory: The thing that we expect to be passed to our
            L{IStreamClientEndpoint.connect} implementation.
        @param connectArgs: Optional dictionary of arguments to
            L{IReactorUNIX.connectUNIX}
        """
        address = UNIXAddress(self.mktemp())
        return (endpoints.UNIXClientEndpoint(reactor, address.name, **connectArgs), (address.name, clientFactory, connectArgs.get('timeout', 30), connectArgs.get('checkPID', 0)), address)