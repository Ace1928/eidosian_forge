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
@skipIf(not sslPresent, 'SSL not present, cannot run SSL tests.')
class AgentHTTPSTests(TestCase, FakeReactorAndConnectMixin, IntegrationTestingMixin):
    """
    Tests for the new HTTP client API that depends on SSL.
    """

    def makeEndpoint(self, host=b'example.com', port=443):
        """
        Create an L{Agent} with an https scheme and return its endpoint
        created according to the arguments.

        @param host: The host for the endpoint.
        @type host: L{bytes}

        @param port: The port for the endpoint.
        @type port: L{int}

        @return: An endpoint of an L{Agent} constructed according to args.
        @rtype: L{SSL4ClientEndpoint}
        """
        return client.Agent(self.createReactor())._getEndpoint(URI.fromBytes(b'https://%b:%d/' % (host, port)))

    def test_endpointType(self):
        """
        L{Agent._getEndpoint} return a L{SSL4ClientEndpoint} when passed a
        scheme of C{'https'}.
        """
        from twisted.internet.endpoints import _WrapperEndpoint
        endpoint = self.makeEndpoint()
        self.assertIsInstance(endpoint, _WrapperEndpoint)
        self.assertIsInstance(endpoint._wrappedEndpoint, HostnameEndpoint)

    def test_hostArgumentIsRespected(self):
        """
        If a host is passed, the endpoint respects it.
        """
        endpoint = self.makeEndpoint(host=b'example.com')
        self.assertEqual(endpoint._wrappedEndpoint._hostStr, 'example.com')

    def test_portArgumentIsRespected(self):
        """
        If a port is passed, the endpoint respects it.
        """
        expectedPort = 4321
        endpoint = self.makeEndpoint(port=expectedPort)
        self.assertEqual(endpoint._wrappedEndpoint._port, expectedPort)

    def test_contextFactoryType(self):
        """
        L{Agent} wraps its connection creator creator and uses modern TLS APIs.
        """
        endpoint = self.makeEndpoint()
        contextFactory = endpoint._wrapperFactory(None)._connectionCreator
        self.assertIsInstance(contextFactory, ClientTLSOptions)
        self.assertEqual(contextFactory._hostname, 'example.com')

    def test_connectHTTPSCustomConnectionCreator(self):
        """
        If a custom L{WebClientConnectionCreator}-like object is passed to
        L{Agent.__init__} it will be used to determine the SSL parameters for
        HTTPS requests.  When an HTTPS request is made, the hostname and port
        number of the request URL will be passed to the connection creator's
        C{creatorForNetloc} method.  The resulting context object will be used
        to establish the SSL connection.
        """
        expectedHost = b'example.org'
        expectedPort = 20443

        class JustEnoughConnection:
            handshakeStarted = False
            connectState = False

            def do_handshake(self):
                """
                The handshake started.  Record that fact.
                """
                self.handshakeStarted = True

            def set_connect_state(self):
                """
                The connection started.  Record that fact.
                """
                self.connectState = True
        contextArgs = []

        @implementer(IOpenSSLClientConnectionCreator)
        class JustEnoughCreator:

            def __init__(self, hostname, port):
                self.hostname = hostname
                self.port = port

            def clientConnectionForTLS(self, tlsProtocol):
                """
                Implement L{IOpenSSLClientConnectionCreator}.

                @param tlsProtocol: The TLS protocol.
                @type tlsProtocol: L{TLSMemoryBIOProtocol}

                @return: C{expectedConnection}
                """
                contextArgs.append((tlsProtocol, self.hostname, self.port))
                return expectedConnection
        expectedConnection = JustEnoughConnection()

        @implementer(IPolicyForHTTPS)
        class StubBrowserLikePolicyForHTTPS:

            def creatorForNetloc(self, hostname, port):
                """
                Emulate L{BrowserLikePolicyForHTTPS}.

                @param hostname: The hostname to verify.
                @type hostname: L{bytes}

                @param port: The port number.
                @type port: L{int}

                @return: a stub L{IOpenSSLClientConnectionCreator}
                @rtype: L{JustEnoughCreator}
                """
                return JustEnoughCreator(hostname, port)
        expectedCreatorCreator = StubBrowserLikePolicyForHTTPS()
        reactor = self.createReactor()
        agent = client.Agent(reactor, expectedCreatorCreator)
        endpoint = agent._getEndpoint(URI.fromBytes(b'https://%b:%d' % (expectedHost, expectedPort)))
        endpoint.connect(Factory.forProtocol(Protocol))
        tlsFactory = reactor.tcpClients[-1][2]
        tlsProtocol = tlsFactory.buildProtocol(None)
        tlsProtocol.makeConnection(StringTransport())
        tls = contextArgs[0][0]
        self.assertIsInstance(tls, TLSMemoryBIOProtocol)
        self.assertEqual(contextArgs[0][1:], (expectedHost, expectedPort))
        self.assertTrue(expectedConnection.handshakeStarted)
        self.assertTrue(expectedConnection.connectState)

    def test_deprecatedDuckPolicy(self):
        """
        Passing something that duck-types I{like} a L{web client context
        factory <twisted.web.client.WebClientContextFactory>} - something that
        does not provide L{IPolicyForHTTPS} - to L{Agent} emits a
        L{DeprecationWarning} even if you don't actually C{import
        WebClientContextFactory} to do it.
        """

        def warnMe():
            client.Agent(deterministicResolvingReactor(MemoryReactorClock()), 'does-not-provide-IPolicyForHTTPS')
        warnMe()
        warnings = self.flushWarnings([warnMe])
        self.assertEqual(len(warnings), 1)
        [warning] = warnings
        self.assertEqual(warning['category'], DeprecationWarning)
        self.assertEqual(warning['message'], "'does-not-provide-IPolicyForHTTPS' was passed as the HTTPS policy for an Agent, but it does not provide IPolicyForHTTPS.  Since Twisted 14.0, you must pass a provider of IPolicyForHTTPS.")

    def test_alternateTrustRoot(self):
        """
        L{BrowserLikePolicyForHTTPS.creatorForNetloc} returns an
        L{IOpenSSLClientConnectionCreator} provider which will add certificates
        from the given trust root.
        """
        trustRoot = CustomOpenSSLTrustRoot()
        policy = BrowserLikePolicyForHTTPS(trustRoot=trustRoot)
        creator = policy.creatorForNetloc(b'thingy', 4321)
        self.assertTrue(trustRoot.called)
        connection = creator.clientConnectionForTLS(None)
        self.assertIs(trustRoot.context, connection.get_context())

    def integrationTest(self, hostName, expectedAddress, addressType):
        """
        Wrap L{AgentTestsMixin.integrationTest} with TLS.
        """
        certHostName = hostName.strip(b'[]')
        authority, server = certificatesForAuthorityAndServer(certHostName.decode('ascii'))

        def tlsify(serverFactory, reactor):
            return TLSMemoryBIOFactory(server.options(), False, serverFactory, reactor)

        def tlsagent(reactor):
            from zope.interface import implementer
            from twisted.web.iweb import IPolicyForHTTPS

            @implementer(IPolicyForHTTPS)
            class Policy:

                def creatorForNetloc(self, hostname, port):
                    return optionsForClientTLS(hostname.decode('ascii'), trustRoot=authority)
            return client.Agent(reactor, contextFactory=Policy())
        super().integrationTest(hostName, expectedAddress, addressType, serverWrapper=tlsify, createAgent=tlsagent, scheme=b'https')