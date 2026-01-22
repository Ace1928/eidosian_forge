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
class _HostnameEndpointMemoryReactorMixin(ClientEndpointTestCaseMixin):
    """
    Common methods for testing L{HostnameEndpoint} against
    L{MemoryReactor} instances that do not provide
    L{IReactorPluggableNameResolver}.
    """

    def synchronousDeferredToThread(self, f, *args, **kwargs):
        """
        A synchronous version of L{deferToThread}.

        @param f: The callable to invoke.
        @type f: L{callable}

        @param args: Positional arguments to the callable.

        @param kwargs: Keyword arguments to the callable.

        @return: A L{Deferred} that fires with the result of applying
            C{f} to C{args} and C{kwargs} or the exception raised.
        """
        try:
            result = f(*args, **kwargs)
        except BaseException:
            return defer.fail()
        else:
            return defer.succeed(result)

    def expectedClients(self, reactor):
        """
        Extract expected clients from the reactor.

        @param reactor: The L{MemoryReactor} under test.

        @return: List of calls to L{IReactorTCP.connectTCP}
        """
        return reactor.tcpClients

    def connectArgs(self):
        """

        @return: C{dict} of keyword arguments to pass to connect.
        """
        return {'timeout': 10.0, 'bindAddress': ('localhost', 49595)}

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

    def test_endpointConnectFailure(self):
        """
        When L{HostnameEndpoint.connect} cannot connect to its
        destination, the returned L{Deferred} will fail with
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

    def test_deprecation(self):
        """
        Instantiating L{HostnameEndpoint} with a reactor that does not
        provide L{IReactorPluggableResolver} emits a deprecation warning.
        """
        mreactor = MemoryReactor()
        clientFactory = object()
        self.createClientEndpoint(mreactor, clientFactory)
        warnings = self.flushWarnings()
        self.assertEqual(1, len(warnings))
        self.assertIs(DeprecationWarning, warnings[0]['category'])
        self.assertTrue(warnings[0]['message'].startswith('Passing HostnameEndpoint a reactor that does not provide IReactorPluggableNameResolver (twisted.internet.testing.MemoryReactorClock) was deprecated in Twisted 17.5.0; please use a reactor that provides IReactorPluggableNameResolver instead'))

    def test_errorsLogged(self):
        """
        Hostname resolution errors are logged.
        """
        mreactor = MemoryReactor()
        clientFactory = object()
        ep, ignoredArgs, ignoredDest = self.createClientEndpoint(mreactor, clientFactory)

        def getaddrinfoThatFails(*args, **kwargs):
            raise gaierror(-5, 'No address associated with hostname')
        ep._getaddrinfo = getaddrinfoThatFails
        d = ep.connect(clientFactory)
        self.assertIsInstance(self.failureResultOf(d).value, error.DNSLookupError)
        self.assertEqual(1, len(self.flushLoggedErrors(gaierror)))