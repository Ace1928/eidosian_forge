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
class WrappedIProtocolTests(unittest.TestCase):
    """
    Test the behaviour of the implementation detail C{_WrapIProtocol}.
    """

    def setUp(self):
        self.reactor = MemoryProcessReactor()
        self.ep = endpoints.ProcessEndpoint(self.reactor, b'/bin/executable')
        self.eventLog = None
        self.factory = protocol.Factory()
        self.factory.protocol = StubApplicationProtocol

    def test_constructor(self):
        """
        Stores an L{IProtocol} provider and the flag to log/drop stderr
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        self.assertIsInstance(wpp.protocol, StubApplicationProtocol)
        self.assertEqual(wpp.errFlag, self.ep._errFlag)

    def test_makeConnection(self):
        """
        Our process transport is properly hooked up to the wrappedIProtocol
        when a connection is made.
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        self.assertEqual(wpp.protocol.transport, wpp.transport)

    def _stdLog(self, eventDict):
        """
        A log observer.
        """
        self.eventLog = eventDict

    def test_logStderr(self):
        """
        When the _errFlag is set to L{StandardErrorBehavior.LOG},
        L{endpoints._WrapIProtocol} logs stderr (in childDataReceived).
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        log.addObserver(self._stdLog)
        self.addCleanup(log.removeObserver, self._stdLog)
        wpp.childDataReceived(2, b'stderr1')
        self.assertEqual(self.eventLog['executable'], wpp.executable)
        self.assertEqual(self.eventLog['data'], b'stderr1')
        self.assertEqual(self.eventLog['protocol'], wpp.protocol)
        self.assertIn('wrote stderr unhandled by', log.textFromEventDict(self.eventLog))

    def test_stderrSkip(self):
        """
        When the _errFlag is set to L{StandardErrorBehavior.DROP},
        L{endpoints._WrapIProtocol} ignores stderr.
        """
        self.ep._errFlag = StandardErrorBehavior.DROP
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        log.addObserver(self._stdLog)
        self.addCleanup(log.removeObserver, self._stdLog)
        wpp.childDataReceived(2, b'stderr2')
        self.assertIsNone(self.eventLog)

    def test_stdout(self):
        """
        In childDataReceived of L{_WrappedIProtocol} instance, the protocol's
        dataReceived is called when stdout is generated.
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        wpp.childDataReceived(1, b'stdout')
        self.assertEqual(wpp.protocol.data, b'stdout')

    def test_processDone(self):
        """
        L{error.ProcessDone} with status=0 is turned into a clean disconnect
        type, i.e. L{error.ConnectionDone}.
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        wpp.processEnded(Failure(error.ProcessDone(0)))
        self.assertEqual(wpp.protocol.reason.check(error.ConnectionDone), error.ConnectionDone)

    def test_processEnded(self):
        """
        Exceptions other than L{error.ProcessDone} with status=0 are turned
        into L{error.ConnectionLost}.
        """
        d = self.ep.connect(self.factory)
        self.successResultOf(d)
        wpp = self.reactor.processProtocol
        wpp.processEnded(Failure(error.ProcessTerminated()))
        self.assertEqual(wpp.protocol.reason.check(error.ConnectionLost), error.ConnectionLost)