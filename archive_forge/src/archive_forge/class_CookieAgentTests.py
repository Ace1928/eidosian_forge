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
class CookieAgentTests(TestCase, CookieTestsMixin, FakeReactorAndConnectMixin, AgentTestsMixin):
    """
    Tests for L{twisted.web.client.CookieAgent}.
    """

    def makeAgent(self):
        """
        @return: a new L{twisted.web.client.CookieAgent}
        """
        return client.CookieAgent(self.buildAgentForWrapperTest(self.reactor), CookieJar())

    def setUp(self):
        self.reactor = self.createReactor()

    def test_emptyCookieJarRequest(self):
        """
        L{CookieAgent.request} does not insert any C{'Cookie'} header into the
        L{Request} object if there is no cookie in the cookie jar for the URI
        being requested. Cookies are extracted from the response and stored in
        the cookie jar.
        """
        cookieJar = CookieJar()
        self.assertEqual(list(cookieJar), [])
        agent = self.buildAgentForWrapperTest(self.reactor)
        cookieAgent = client.CookieAgent(agent, cookieJar)
        d = cookieAgent.request(b'GET', b'http://example.com:1234/foo?bar')

        def _checkCookie(ignored):
            cookies = list(cookieJar)
            self.assertEqual(len(cookies), 1)
            self.assertEqual(cookies[0].name, 'foo')
            self.assertEqual(cookies[0].value, '1')
        d.addCallback(_checkCookie)
        req, res = self.protocol.requests.pop()
        self.assertIdentical(req.headers.getRawHeaders(b'cookie'), None)
        resp = client.Response((b'HTTP', 1, 1), 200, b'OK', Headers({b'Set-Cookie': [b'foo=1']}), None)
        res.callback(resp)
        return d

    def test_leaveExistingCookieHeader(self) -> None:
        """
        L{CookieAgent.request} will not insert a C{'Cookie'} header into the
        L{Request} object when there is already a C{'Cookie'} header in the
        request headers parameter.
        """
        uri = b'http://example.com:1234/foo?bar'
        cookie = b'foo=1'
        cookieJar = CookieJar()
        self.addCookies(cookieJar, uri, [cookie])
        self.assertEqual(len(list(cookieJar)), 1)
        agent = self.buildAgentForWrapperTest(self.reactor)
        cookieAgent = client.CookieAgent(agent, cookieJar)
        cookieAgent.request(b'GET', uri, Headers({'cookie': ['already-set']}))
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'cookie'), [b'already-set'])

    def test_requestWithCookie(self):
        """
        L{CookieAgent.request} inserts a C{'Cookie'} header into the L{Request}
        object when there is a cookie matching the request URI in the cookie
        jar.
        """
        uri = b'http://example.com:1234/foo?bar'
        cookie = b'foo=1'
        cookieJar = CookieJar()
        self.addCookies(cookieJar, uri, [cookie])
        self.assertEqual(len(list(cookieJar)), 1)
        agent = self.buildAgentForWrapperTest(self.reactor)
        cookieAgent = client.CookieAgent(agent, cookieJar)
        cookieAgent.request(b'GET', uri)
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'cookie'), [cookie])

    @skipIf(not sslPresent, 'SSL not present, cannot run SSL tests.')
    def test_secureCookie(self):
        """
        L{CookieAgent} is able to handle secure cookies, ie cookies which
        should only be handled over https.
        """
        uri = b'https://example.com:1234/foo?bar'
        cookie = b'foo=1;secure'
        cookieJar = CookieJar()
        self.addCookies(cookieJar, uri, [cookie])
        self.assertEqual(len(list(cookieJar)), 1)
        agent = self.buildAgentForWrapperTest(self.reactor)
        cookieAgent = client.CookieAgent(agent, cookieJar)
        cookieAgent.request(b'GET', uri)
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'cookie'), [b'foo=1'])

    def test_secureCookieOnInsecureConnection(self):
        """
        If a cookie is setup as secure, it won't be sent with the request if
        it's not over HTTPS.
        """
        uri = b'http://example.com/foo?bar'
        cookie = b'foo=1;secure'
        cookieJar = CookieJar()
        self.addCookies(cookieJar, uri, [cookie])
        self.assertEqual(len(list(cookieJar)), 1)
        agent = self.buildAgentForWrapperTest(self.reactor)
        cookieAgent = client.CookieAgent(agent, cookieJar)
        cookieAgent.request(b'GET', uri)
        req, res = self.protocol.requests.pop()
        self.assertIdentical(None, req.headers.getRawHeaders(b'cookie'))

    def test_portCookie(self):
        """
        L{CookieAgent} supports cookies which enforces the port number they
        need to be transferred upon.
        """
        uri = b'http://example.com:1234/foo?bar'
        cookie = b'foo=1;port=1234'
        cookieJar = CookieJar()
        self.addCookies(cookieJar, uri, [cookie])
        self.assertEqual(len(list(cookieJar)), 1)
        agent = self.buildAgentForWrapperTest(self.reactor)
        cookieAgent = client.CookieAgent(agent, cookieJar)
        cookieAgent.request(b'GET', uri)
        req, res = self.protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'cookie'), [b'foo=1'])

    def test_portCookieOnWrongPort(self):
        """
        When creating a cookie with a port directive, it won't be added to the
        L{cookie.CookieJar} if the URI is on a different port.
        """
        uri = b'http://example.com:4567/foo?bar'
        cookie = b'foo=1;port=1234'
        cookieJar = CookieJar()
        self.addCookies(cookieJar, uri, [cookie])
        self.assertEqual(len(list(cookieJar)), 0)