from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
class IntegrationTestingMixin:
    """
    Transport-to-Agent integration tests for both HTTP and HTTPS.
    """

    def test_integrationTestIPv4(self):
        """
        L{Agent} works over IPv4.
        """
        self.integrationTest(b'example.com', EXAMPLE_COM_IP, IPv4Address)

    def test_integrationTestIPv4Address(self):
        """
        L{Agent} works over IPv4 when hostname is an IPv4 address.
        """
        self.integrationTest(b'127.0.0.7', '127.0.0.7', IPv4Address)

    def test_integrationTestIPv6(self):
        """
        L{Agent} works over IPv6.
        """
        self.integrationTest(b'ipv6.example.com', EXAMPLE_COM_V6_IP, IPv6Address)

    def test_integrationTestIPv6Address(self):
        """
        L{Agent} works over IPv6 when hostname is an IPv6 address.
        """
        self.integrationTest(b'[::7]', '::7', IPv6Address)

    def integrationTest(self, hostName, expectedAddress, addressType, serverWrapper=lambda server, _: server, createAgent=client.Agent, scheme=b'http'):
        """
        L{Agent} will make a TCP connection, send an HTTP request, and return a
        L{Deferred} that fires when the response has been received.

        @param hostName: The hostname to interpolate into the URL to be
            requested.
        @type hostName: L{bytes}

        @param expectedAddress: The expected address string.
        @type expectedAddress: L{bytes}

        @param addressType: The class to construct an address out of.
        @type addressType: L{type}

        @param serverWrapper: A callable that takes a protocol factory and a
            ``Clock`` and returns a protocol factory; used to wrap the server /
            responder side in a TLS server.
        @type serverWrapper:
            serverWrapper(L{twisted.internet.interfaces.IProtocolFactory}) ->
            L{twisted.internet.interfaces.IProtocolFactory}

        @param createAgent: A callable that takes a reactor and produces an
            L{IAgent}; used to construct an agent with an appropriate trust
            root for TLS.
        @type createAgent: createAgent(reactor) -> L{IAgent}

        @param scheme: The scheme to test, C{http} or C{https}
        @type scheme: L{bytes}
        """
        reactor = self.createReactor()
        if sslPresent:
            self.patch(tls, '_get_default_clock', lambda: reactor)
        agent = createAgent(reactor)
        deferred = agent.request(b'GET', scheme + b'://' + hostName + b'/')
        host, port, factory, timeout, bind = reactor.tcpClients[0]
        self.assertEqual(host, expectedAddress)
        peerAddress = addressType('TCP', host, port)
        clientProtocol = factory.buildProtocol(peerAddress)
        clientTransport = FakeTransport(clientProtocol, False, peerAddress=peerAddress)
        clientProtocol.makeConnection(clientTransport)

        @Factory.forProtocol
        def accumulator():
            ap = AccumulatingProtocol()
            accumulator.currentProtocol = ap
            return ap
        accumulator.currentProtocol = None
        accumulator.protocolConnectionMade = None
        wrapper = serverWrapper(accumulator, reactor).buildProtocol(None)
        serverTransport = FakeTransport(wrapper, True)
        wrapper.makeConnection(serverTransport)
        pump = IOPump(clientProtocol, wrapper, clientTransport, serverTransport, False, clock=reactor)
        pump.flush()
        self.assertNoResult(deferred)
        lines = accumulator.currentProtocol.data.split(b'\r\n')
        self.assertTrue(lines[0].startswith(b'GET / HTTP'), lines[0])
        headers = dict([line.split(b': ', 1) for line in lines[1:] if line])
        self.assertEqual(headers[b'Host'], hostName)
        self.assertNoResult(deferred)
        accumulator.currentProtocol.transport.write(b'HTTP/1.1 200 OK\r\nX-An-Header: an-value\r\n\r\nContent-length: 12\r\n\r\nhello world!')
        pump.flush()
        response = self.successResultOf(deferred)
        self.assertEquals(response.headers.getRawHeaders(b'x-an-header')[0], b'an-value')