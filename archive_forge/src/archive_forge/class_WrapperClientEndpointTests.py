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
class WrapperClientEndpointTests(unittest.TestCase):
    """
    Tests for L{_WrapperClientEndpoint}.
    """

    def setUp(self):
        self.endpoint, self.completer = connectableEndpoint()
        self.context = object()
        self.wrapper = endpoints._WrapperEndpoint(self.endpoint, UppercaseWrapperFactory)
        self.factory = Factory.forProtocol(NetstringTracker)

    def test_wrappingBehavior(self):
        """
        Any modifications performed by the underlying L{ProtocolWrapper}
        propagate through to the wrapped L{Protocol}.
        """
        connecting = self.wrapper.connect(self.factory)
        pump = self.completer.succeedOnce()
        proto = self.successResultOf(connecting)
        pump.server.transport.write(b'5:hello,')
        pump.flush()
        self.assertEqual(proto.strings, [b'HELLO'])

    def test_methodsAvailable(self):
        """
        Methods defined on the wrapped L{Protocol} are accessible from the
        L{Protocol} returned from C{connect}'s L{Deferred}.
        """
        connecting = self.wrapper.connect(self.factory)
        pump = self.completer.succeedOnce()
        proto = self.successResultOf(connecting)
        proto.sendString(b'spam')
        self.assertEqual(pump.clientIO.getOutBuffer(), b'4:SPAM,')

    def test_connectionFailure(self):
        """
        Connection failures propagate upward to C{connect}'s L{Deferred}.
        """
        d = self.wrapper.connect(self.factory)
        self.assertNoResult(d)
        self.completer.failOnce(FakeError())
        self.failureResultOf(d, FakeError)

    def test_connectionCancellation(self):
        """
        Cancellation propagates upward to C{connect}'s L{Deferred}.
        """
        d = self.wrapper.connect(self.factory)
        self.assertNoResult(d)
        d.cancel()
        self.failureResultOf(d, ConnectingCancelledError)

    def test_transportOfTransportOfWrappedProtocol(self):
        """
        The transport of the wrapped L{Protocol}'s transport is the transport
        passed to C{makeConnection}.
        """
        connecting = self.wrapper.connect(self.factory)
        pump = self.completer.succeedOnce()
        proto = self.successResultOf(connecting)
        self.assertIs(proto.transport.transport, pump.clientIO)