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
class HostnameEndpointIDNATests(unittest.SynchronousTestCase):
    """
    Tests for L{HostnameEndpoint}'s constructor's encoding behavior.
    """
    sampleIDNAText = 'bücher.ch'
    sampleIDNABytes = b'xn--bcher-kva.ch'

    def test_idnaHostnameText(self):
        """
        A L{HostnameEndpoint} constructed with text will contain an
        IDNA-encoded bytes representation of that text.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(MemoryReactor(), ['127.0.0.1']), self.sampleIDNAText, 80)
        self.assertEqual(endpoint._hostBytes, self.sampleIDNABytes)
        self.assertEqual(endpoint._hostText, self.sampleIDNAText)

    def test_idnaHostnameBytes(self):
        """
        A L{HostnameEndpoint} constructed with bytes will contain an
        IDNA-decoded textual representation of those bytes.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(MemoryReactor(), ['127.0.0.1']), self.sampleIDNAText, 80)
        self.assertEqual(endpoint._hostBytes, self.sampleIDNABytes)
        self.assertEqual(endpoint._hostText, self.sampleIDNAText)

    def test_nonNormalizedText(self):
        """
        A L{HostnameEndpoint} constructed with NFD-normalized text will store
        the NFC-normalized version of that text.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(MemoryReactor(), ['127.0.0.1']), normalize('NFD', self.sampleIDNAText), 80)
        self.assertEqual(endpoint._hostBytes, self.sampleIDNABytes)
        self.assertEqual(endpoint._hostText, self.sampleIDNAText)

    def test_deferBadEncodingToConnect(self):
        """
        Since any client of L{IStreamClientEndpoint} needs to handle Deferred
        failures from C{connect}, L{HostnameEndpoint}'s constructor will not
        raise exceptions when given bad host names, instead deferring to
        returning a failing L{Deferred} from C{connect}.
        """
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(MemoryReactor(), ['127.0.0.1']), b'\xff-garbage-\xff', 80)
        deferred = endpoint.connect(Factory.forProtocol(Protocol))
        err = self.failureResultOf(deferred, ValueError)
        self.assertIn('\\xff-garbage-\\xff', str(err))
        endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(MemoryReactor(), ['127.0.0.1']), '⿰-garbage-⿰', 80)
        deferred = endpoint.connect(Factory())
        err = self.failureResultOf(deferred, ValueError)
        self.assertIn('\\u2ff0-garbage-\\u2ff0', str(err))