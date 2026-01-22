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
class ContentDecoderAgentTests(TestCase, FakeReactorAndConnectMixin, AgentTestsMixin):
    """
    Tests for L{client.ContentDecoderAgent}.
    """

    def makeAgent(self):
        """
        @return: a new L{twisted.web.client.ContentDecoderAgent}
        """
        return client.ContentDecoderAgent(self.agent, [])

    def setUp(self):
        """
        Create an L{Agent} wrapped around a fake reactor.
        """
        self.reactor = self.createReactor()
        self.agent = self.buildAgentForWrapperTest(self.reactor)

    def test_acceptHeaders(self):
        """
        L{client.ContentDecoderAgent} sets the I{Accept-Encoding} header to the
        names of the available decoder objects.
        """
        agent = client.ContentDecoderAgent(self.agent, [(b'decoder1', Decoder1), (b'decoder2', Decoder2)])
        agent.request(b'GET', b'http://example.com/foo')
        protocol = self.protocol
        self.assertEqual(len(protocol.requests), 1)
        req, res = protocol.requests.pop()
        self.assertEqual(req.headers.getRawHeaders(b'accept-encoding'), [b'decoder1,decoder2'])

    def test_existingHeaders(self):
        """
        If there are existing I{Accept-Encoding} fields,
        L{client.ContentDecoderAgent} creates a new field for the decoders it
        knows about.
        """
        headers = http_headers.Headers({b'foo': [b'bar'], b'accept-encoding': [b'fizz']})
        agent = client.ContentDecoderAgent(self.agent, [(b'decoder1', Decoder1), (b'decoder2', Decoder2)])
        agent.request(b'GET', b'http://example.com/foo', headers=headers)
        protocol = self.protocol
        self.assertEqual(len(protocol.requests), 1)
        req, res = protocol.requests.pop()
        self.assertEqual(list(sorted(req.headers.getAllRawHeaders())), [(b'Accept-Encoding', [b'fizz', b'decoder1,decoder2']), (b'Foo', [b'bar']), (b'Host', [b'example.com'])])

    def test_plainEncodingResponse(self):
        """
        If the response is not encoded despited the request I{Accept-Encoding}
        headers, L{client.ContentDecoderAgent} simply forwards the response.
        """
        agent = client.ContentDecoderAgent(self.agent, [(b'decoder1', Decoder1), (b'decoder2', Decoder2)])
        deferred = agent.request(b'GET', b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        response = Response((b'HTTP', 1, 1), 200, b'OK', http_headers.Headers(), None)
        res.callback(response)
        return deferred.addCallback(self.assertIdentical, response)

    def test_unsupportedEncoding(self):
        """
        If an encoding unknown to the L{client.ContentDecoderAgent} is found,
        the response is unchanged.
        """
        agent = client.ContentDecoderAgent(self.agent, [(b'decoder1', Decoder1), (b'decoder2', Decoder2)])
        deferred = agent.request(b'GET', b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers({b'foo': [b'bar'], b'content-encoding': [b'fizz']})
        response = Response((b'HTTP', 1, 1), 200, b'OK', headers, None)
        res.callback(response)
        return deferred.addCallback(self.assertIdentical, response)

    def test_unknownEncoding(self):
        """
        When L{client.ContentDecoderAgent} encounters a decoder it doesn't know
        about, it stops decoding even if another encoding is known afterwards.
        """
        agent = client.ContentDecoderAgent(self.agent, [(b'decoder1', Decoder1), (b'decoder2', Decoder2)])
        deferred = agent.request(b'GET', b'http://example.com/foo')
        req, res = self.protocol.requests.pop()
        headers = http_headers.Headers({b'foo': [b'bar'], b'content-encoding': [b'decoder1,fizz,decoder2']})
        response = Response((b'HTTP', 1, 1), 200, b'OK', headers, None)
        res.callback(response)

        def check(result):
            self.assertNotIdentical(response, result)
            self.assertIsInstance(result, Decoder2)
            self.assertEqual([b'decoder1,fizz'], result.headers.getRawHeaders(b'content-encoding'))
        return deferred.addCallback(check)