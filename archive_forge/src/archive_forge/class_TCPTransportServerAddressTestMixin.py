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
class TCPTransportServerAddressTestMixin:
    """
    Test mixing for TCP server address building and log prefix.
    """

    def getConnectedClientAndServer(self, reactor, interface, addressFamily):
        """
        Helper method returnine a L{Deferred} firing with a tuple of a client
        protocol, a server protocol, and a running TCP port.
        """
        raise NotImplementedError()

    def _testServerAddress(self, interface, addressFamily, adressClass):
        """
        Helper method to test TCP server addresses on either IPv4 or IPv6.
        """

        def connected(protocols):
            client, server, port = protocols
            try:
                self.assertEqual('<AccumulatingProtocol #%s on %s>' % (server.transport.sessionno, port.getHost().port), str(server.transport))
                self.assertEqual('AccumulatingProtocol,%s,%s' % (server.transport.sessionno, interface), server.transport.logstr)
                [peerAddress] = server.factory.peerAddresses
                self.assertIsInstance(peerAddress, adressClass)
                self.assertEqual('TCP', peerAddress.type)
                self.assertEqual(interface, peerAddress.host)
            finally:
                server.transport.loseConnection()
        reactor = self.buildReactor()
        d = self.getConnectedClientAndServer(reactor, interface, addressFamily)
        d.addCallback(connected)
        d.addErrback(log.err)
        self.runReactor(reactor)

    def test_serverAddressTCP4(self):
        """
        L{Server} instances have a string representation indicating on which
        port they're running, and the connected address is stored on the
        C{peerAddresses} attribute of the factory.
        """
        return self._testServerAddress('127.0.0.1', socket.AF_INET, IPv4Address)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_serverAddressTCP6(self):
        """
        IPv6 L{Server} instances have a string representation indicating on
        which port they're running, and the connected address is stored on the
        C{peerAddresses} attribute of the factory.
        """
        return self._testServerAddress(getLinkLocalIPv6Address(), socket.AF_INET6, IPv6Address)