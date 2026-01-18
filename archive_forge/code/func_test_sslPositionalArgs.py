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
def test_sslPositionalArgs(self):
    """
        When passed an SSL strports description, L{clientFromString} returns a
        L{SSL4ClientEndpoint} instance initialized with the values from the
        string.
        """
    reactor = object()
    client = endpoints.clientFromString(reactor, 'ssl:example.net:4321:privateKey=%s:certKey=%s:bindAddress=10.0.0.3:timeout=3:caCertsDir=%s' % (escapedPEMPathName, escapedPEMPathName, escapedCAsPathName))
    self.assertIsInstance(client, endpoints.SSL4ClientEndpoint)
    self.assertIs(client._reactor, reactor)
    self.assertEqual(client._host, 'example.net')
    self.assertEqual(client._port, 4321)
    self.assertEqual(client._timeout, 3)
    self.assertEqual(client._bindAddress, ('10.0.0.3', 0))