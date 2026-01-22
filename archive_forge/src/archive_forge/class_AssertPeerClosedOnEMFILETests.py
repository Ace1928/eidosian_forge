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
@skipIf(SKIP_EMFILE, 'Reserved EMFILE file descriptor not supported on Windows.')
class AssertPeerClosedOnEMFILETests(SynchronousTestCase):
    """
    Tests for L{assertPeerClosedOnEMFILE}.
    """

    @implementer(_IExhaustsFileDescriptors)
    class NullExhauster:
        """
        An exhauster that does nothing.
        """

        def exhaust(self):
            """
            See L{_IExhaustsFileDescriptors.exhaust}
            """

        def release(self):
            """
            See L{_IExhaustsFileDescriptors.release}
            """

        def count(self):
            """
            See L{_IExhaustsFileDescriptors.count}
            """

    def setUp(self):
        self.reactor = MemoryReactor()
        self.testCase = SynchronousTestCase()

    def test_nullExhausterProvidesInterface(self):
        """
        L{NullExhauster} instances provide
        L{_IExhaustsFileDescriptors}.
        """
        verifyObject(_IExhaustsFileDescriptors, self.NullExhauster())

    def test_reactorStoppedOnSuccessfulConnection(self):
        """
        If the exhauster fails to trigger C{EMFILE} and a connection
        reaches the server, the reactor is stopped and the test fails.
        """
        exhauster = self.NullExhauster()
        serverFactory = [None]

        def runReactor(reactor):
            reactor.run()
            proto = serverFactory[0].buildProtocol(IPv4Address('TCP', '127.0.0.1', 4321))
            proto.makeConnection(StringTransport())

        def listen(reactor, factory):
            port = reactor.listenTCP('127.0.0.1', 1234, factory)
            factory.doStart()
            serverFactory[0] = factory
            return port

        def connect(reactor, address, factory):
            reactor.connectTCP('127.0.0.1', 0, factory)
        exception = self.assertRaises(self.testCase.failureException, assertPeerClosedOnEMFILE, testCase=self.testCase, exhauster=exhauster, reactor=self.reactor, runReactor=runReactor, listen=listen, connect=connect)
        self.assertIn('EMFILE', str(exception))
        self.assertFalse(self.reactor.running)