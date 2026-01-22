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
class TCPPortTestsMixin:
    """
    Tests for L{IReactorTCP.listenTCP}
    """
    requiredInterfaces: Optional[Sequence[Type[Interface]]] = (IReactorTCP,)

    def getExpectedStartListeningLogMessage(self, port, factory):
        """
        Get the message expected to be logged when a TCP port starts listening.
        """
        return '%s starting on %d' % (factory, port.getHost().port)

    def getExpectedConnectionLostLogMsg(self, port):
        """
        Get the expected connection lost message for a TCP port.
        """
        return f'(TCP Port {port.getHost().port} Closed)'

    def test_portGetHostOnIPv4(self):
        """
        When no interface is passed to L{IReactorTCP.listenTCP}, the returned
        listening port listens on an IPv4 address.
        """
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, ServerFactory())
        address = port.getHost()
        self.assertIsInstance(address, IPv4Address)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_portGetHostOnIPv6(self):
        """
        When listening on an IPv6 address, L{IListeningPort.getHost} returns
        an L{IPv6Address} with C{host} and C{port} attributes reflecting the
        address the port is bound to.
        """
        reactor = self.buildReactor()
        host, portNumber = findFreePort(family=socket.AF_INET6, interface='::1')[:2]
        port = self.getListeningPort(reactor, ServerFactory(), portNumber, host)
        address = port.getHost()
        self.assertIsInstance(address, IPv6Address)
        self.assertEqual('::1', address.host)
        self.assertEqual(portNumber, address.port)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_portGetHostOnIPv6ScopeID(self):
        """
        When a link-local IPv6 address including a scope identifier is passed
        as the C{interface} argument to L{IReactorTCP.listenTCP}, the resulting
        L{IListeningPort} reports its address as an L{IPv6Address} with a host
        value that includes the scope identifier.
        """
        linkLocal = getLinkLocalIPv6Address()
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, ServerFactory(), 0, linkLocal)
        address = port.getHost()
        self.assertIsInstance(address, IPv6Address)
        self.assertEqual(linkLocal, address.host)

    def _buildProtocolAddressTest(self, client, interface):
        """
        Connect C{client} to a server listening on C{interface} started with
        L{IReactorTCP.listenTCP} and return the address passed to the factory's
        C{buildProtocol} method.

        @param client: A C{SOCK_STREAM} L{socket.socket} created with an address
            family such that it will be able to connect to a server listening on
            C{interface}.

        @param interface: A C{str} giving an address for a server to listen on.
            This should almost certainly be the loopback address for some
            address family supported by L{IReactorTCP.listenTCP}.

        @return: Whatever object, probably an L{IAddress} provider, is passed to
            a server factory's C{buildProtocol} method when C{client}
            establishes a connection.
        """

        class ObserveAddress(ServerFactory):

            def buildProtocol(self, address):
                reactor.stop()
                self.observedAddress = address
                return Protocol()
        factory = ObserveAddress()
        reactor = self.buildReactor()
        port = self.getListeningPort(reactor, factory, 0, interface)
        client.setblocking(False)
        try:
            connect(client, (port.getHost().host, port.getHost().port))
        except OSError as e:
            self.assertIn(e.errno, (errno.EINPROGRESS, errno.EWOULDBLOCK))
        self.runReactor(reactor)
        return factory.observedAddress

    def test_buildProtocolIPv4Address(self):
        """
        When a connection is accepted over IPv4, an L{IPv4Address} is passed
        to the factory's C{buildProtocol} method giving the peer's address.
        """
        interface = '127.0.0.1'
        client = createTestSocket(self, socket.AF_INET, socket.SOCK_STREAM)
        observedAddress = self._buildProtocolAddressTest(client, interface)
        self.assertEqual(IPv4Address('TCP', *client.getsockname()), observedAddress)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_buildProtocolIPv6Address(self):
        """
        When a connection is accepted to an IPv6 address, an L{IPv6Address} is
        passed to the factory's C{buildProtocol} method giving the peer's
        address.
        """
        interface = '::1'
        client = createTestSocket(self, socket.AF_INET6, socket.SOCK_STREAM)
        observedAddress = self._buildProtocolAddressTest(client, interface)
        peer = client.getsockname()
        hostname = socket.getnameinfo(peer, socket.NI_NUMERICHOST)[0]
        self.assertEqual(IPv6Address('TCP', hostname, peer[1]), observedAddress)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_buildProtocolIPv6AddressScopeID(self):
        """
        When a connection is accepted to a link-local IPv6 address, an
        L{IPv6Address} is passed to the factory's C{buildProtocol} method
        giving the peer's address, including a scope identifier.
        """
        interface = getLinkLocalIPv6Address()
        client = createTestSocket(self, socket.AF_INET6, socket.SOCK_STREAM)
        observedAddress = self._buildProtocolAddressTest(client, interface)
        peer = client.getsockname()
        hostname = socket.getnameinfo(peer, socket.NI_NUMERICHOST)[0]
        self.assertEqual(IPv6Address('TCP', hostname, *peer[1:]), observedAddress)

    def _serverGetConnectionAddressTest(self, client, interface, which):
        """
        Connect C{client} to a server listening on C{interface} started with
        L{IReactorTCP.listenTCP} and return the address returned by one of the
        server transport's address lookup methods, C{getHost} or C{getPeer}.

        @param client: A C{SOCK_STREAM} L{socket.socket} created with an address
            family such that it will be able to connect to a server listening on
            C{interface}.

        @param interface: A C{str} giving an address for a server to listen on.
            This should almost certainly be the loopback address for some
            address family supported by L{IReactorTCP.listenTCP}.

        @param which: A C{str} equal to either C{"getHost"} or C{"getPeer"}
            determining which address will be returned.

        @return: Whatever object, probably an L{IAddress} provider, is returned
            from the method indicated by C{which}.
        """

        class ObserveAddress(Protocol):

            def makeConnection(self, transport):
                reactor.stop()
                self.factory.address = getattr(transport, which)()
        reactor = self.buildReactor()
        factory = ServerFactory()
        factory.protocol = ObserveAddress
        port = self.getListeningPort(reactor, factory, 0, interface)
        client.setblocking(False)
        try:
            connect(client, (port.getHost().host, port.getHost().port))
        except OSError as e:
            self.assertIn(e.errno, (errno.EINPROGRESS, errno.EWOULDBLOCK))
        self.runReactor(reactor)
        return factory.address

    def test_serverGetHostOnIPv4(self):
        """
        When a connection is accepted over IPv4, the server
        L{ITransport.getHost} method returns an L{IPv4Address} giving the
        address on which the server accepted the connection.
        """
        interface = '127.0.0.1'
        client = createTestSocket(self, socket.AF_INET, socket.SOCK_STREAM)
        hostAddress = self._serverGetConnectionAddressTest(client, interface, 'getHost')
        self.assertEqual(IPv4Address('TCP', *client.getpeername()), hostAddress)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_serverGetHostOnIPv6(self):
        """
        When a connection is accepted over IPv6, the server
        L{ITransport.getHost} method returns an L{IPv6Address} giving the
        address on which the server accepted the connection.
        """
        interface = '::1'
        client = createTestSocket(self, socket.AF_INET6, socket.SOCK_STREAM)
        hostAddress = self._serverGetConnectionAddressTest(client, interface, 'getHost')
        peer = client.getpeername()
        hostname = socket.getnameinfo(peer, socket.NI_NUMERICHOST)[0]
        self.assertEqual(IPv6Address('TCP', hostname, *peer[1:]), hostAddress)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_serverGetHostOnIPv6ScopeID(self):
        """
        When a connection is accepted over IPv6, the server
        L{ITransport.getHost} method returns an L{IPv6Address} giving the
        address on which the server accepted the connection, including the scope
        identifier.
        """
        interface = getLinkLocalIPv6Address()
        client = createTestSocket(self, socket.AF_INET6, socket.SOCK_STREAM)
        hostAddress = self._serverGetConnectionAddressTest(client, interface, 'getHost')
        peer = client.getpeername()
        hostname = socket.getnameinfo(peer, socket.NI_NUMERICHOST)[0]
        self.assertEqual(IPv6Address('TCP', hostname, *peer[1:]), hostAddress)

    def test_serverGetPeerOnIPv4(self):
        """
        When a connection is accepted over IPv4, the server
        L{ITransport.getPeer} method returns an L{IPv4Address} giving the
        address of the remote end of the connection.
        """
        interface = '127.0.0.1'
        client = createTestSocket(self, socket.AF_INET, socket.SOCK_STREAM)
        peerAddress = self._serverGetConnectionAddressTest(client, interface, 'getPeer')
        self.assertEqual(IPv4Address('TCP', *client.getsockname()), peerAddress)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_serverGetPeerOnIPv6(self):
        """
        When a connection is accepted over IPv6, the server
        L{ITransport.getPeer} method returns an L{IPv6Address} giving the
        address on the remote end of the connection.
        """
        interface = '::1'
        client = createTestSocket(self, socket.AF_INET6, socket.SOCK_STREAM)
        peerAddress = self._serverGetConnectionAddressTest(client, interface, 'getPeer')
        peer = client.getsockname()
        hostname = socket.getnameinfo(peer, socket.NI_NUMERICHOST)[0]
        self.assertEqual(IPv6Address('TCP', hostname, *peer[1:]), peerAddress)

    @skipIf(ipv6Skip, ipv6SkipReason)
    def test_serverGetPeerOnIPv6ScopeID(self):
        """
        When a connection is accepted over IPv6, the server
        L{ITransport.getPeer} method returns an L{IPv6Address} giving the
        address on the remote end of the connection, including the scope
        identifier.
        """
        interface = getLinkLocalIPv6Address()
        client = createTestSocket(self, socket.AF_INET6, socket.SOCK_STREAM)
        peerAddress = self._serverGetConnectionAddressTest(client, interface, 'getPeer')
        peer = client.getsockname()
        hostname = socket.getnameinfo(peer, socket.NI_NUMERICHOST)[0]
        self.assertEqual(IPv6Address('TCP', hostname, *peer[1:]), peerAddress)