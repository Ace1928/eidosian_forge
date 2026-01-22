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
class ClientStringTests(unittest.TestCase):
    """
    Tests for L{twisted.internet.endpoints.clientFromString}.
    """

    def test_tcp(self):
        """
        When passed a TCP strports description, L{endpoints.clientFromString}
        returns a L{TCP4ClientEndpoint} instance initialized with the values
        from the string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'tcp:host=example.com:port=1234:timeout=7:bindAddress=10.0.0.2')
        self.assertIsInstance(client, endpoints.TCP4ClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._host, 'example.com')
        self.assertEqual(client._port, 1234)
        self.assertEqual(client._timeout, 7)
        self.assertEqual(client._bindAddress, ('10.0.0.2', 0))

    def test_tcpPositionalArgs(self):
        """
        When passed a TCP strports description using positional arguments,
        L{endpoints.clientFromString} returns a L{TCP4ClientEndpoint} instance
        initialized with the values from the string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'tcp:example.com:1234:timeout=7:bindAddress=10.0.0.2')
        self.assertIsInstance(client, endpoints.TCP4ClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._host, 'example.com')
        self.assertEqual(client._port, 1234)
        self.assertEqual(client._timeout, 7)
        self.assertEqual(client._bindAddress, ('10.0.0.2', 0))

    def test_tcpHostPositionalArg(self):
        """
        When passed a TCP strports description specifying host as a positional
        argument, L{endpoints.clientFromString} returns a L{TCP4ClientEndpoint}
        instance initialized with the values from the string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'tcp:example.com:port=1234:timeout=7:bindAddress=10.0.0.2')
        self.assertEqual(client._host, 'example.com')
        self.assertEqual(client._port, 1234)

    def test_tcpPortPositionalArg(self):
        """
        When passed a TCP strports description specifying port as a positional
        argument, L{endpoints.clientFromString} returns a L{TCP4ClientEndpoint}
        instance initialized with the values from the string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'tcp:host=example.com:1234:timeout=7:bindAddress=10.0.0.2')
        self.assertEqual(client._host, 'example.com')
        self.assertEqual(client._port, 1234)

    def test_tcpDefaults(self):
        """
        A TCP strports description may omit I{timeout} or I{bindAddress} to
        allow the default to be used.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'tcp:host=example.com:port=1234')
        self.assertEqual(client._timeout, 30)
        self.assertIsNone(client._bindAddress)

    def test_unix(self):
        """
        When passed a UNIX strports description, L{endpoints.clientFromString}
        returns a L{UNIXClientEndpoint} instance initialized with the values
        from the string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'unix:path=/var/foo/bar:lockfile=1:timeout=9')
        self.assertIsInstance(client, endpoints.UNIXClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._path, '/var/foo/bar')
        self.assertEqual(client._timeout, 9)
        self.assertTrue(client._checkPID)

    def test_unixDefaults(self):
        """
        A UNIX strports description may omit I{lockfile} or I{timeout} to allow
        the defaults to be used.
        """
        client = endpoints.clientFromString(object(), 'unix:path=/var/foo/bar')
        self.assertEqual(client._timeout, 30)
        self.assertFalse(client._checkPID)

    def test_unixPathPositionalArg(self):
        """
        When passed a UNIX strports description specifying path as a positional
        argument, L{endpoints.clientFromString} returns a L{UNIXClientEndpoint}
        instance initialized with the values from the string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'unix:/var/foo/bar:lockfile=1:timeout=9')
        self.assertIsInstance(client, endpoints.UNIXClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._path, '/var/foo/bar')
        self.assertEqual(client._timeout, 9)
        self.assertTrue(client._checkPID)

    def test_typeFromPlugin(self):
        """
        L{endpoints.clientFromString} looks up plugins of type
        L{IStreamClientEndpoint} and constructs endpoints from them.
        """
        addFakePlugin(self)
        notAReactor = object()
        clientEndpoint = endpoints.clientFromString(notAReactor, 'crfake:alpha:beta:cee=dee:num=1')
        from twisted.plugins.fakeendpoint import fakeClientWithReactor
        self.assertIs(clientEndpoint.parser, fakeClientWithReactor)
        self.assertEqual(clientEndpoint.args, (notAReactor, 'alpha', 'beta'))
        self.assertEqual(clientEndpoint.kwargs, dict(cee='dee', num='1'))

    def test_unknownType(self):
        """
        L{endpoints.clientFromString} raises C{ValueError} when given an
        unknown endpoint type.
        """
        value = self.assertRaises(ValueError, endpoints.clientFromString, None, 'ftl:andromeda/carcosa/hali/2387')
        self.assertEqual(str(value), "Unknown endpoint type: 'ftl'")

    def test_stringParserWithReactor(self):
        """
        L{endpoints.clientFromString} will pass a reactor to plugins
        implementing the L{IStreamClientEndpointStringParserWithReactor}
        interface.
        """
        addFakePlugin(self)
        reactor = object()
        clientEndpoint = endpoints.clientFromString(reactor, 'crfake:alpha:beta:cee=dee:num=1')
        from twisted.plugins.fakeendpoint import fakeClientWithReactor
        self.assertEqual((clientEndpoint.parser, clientEndpoint.args, clientEndpoint.kwargs), (fakeClientWithReactor, (reactor, 'alpha', 'beta'), dict(cee='dee', num='1')))