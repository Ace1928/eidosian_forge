from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
class ProxyClientFactoryTests(TestCase):
    """
    Tests for L{ProxyClientFactory}.
    """

    def test_connectionFailed(self):
        """
        Check that L{ProxyClientFactory.clientConnectionFailed} produces
        a B{501} response to the parent request.
        """
        request = DummyRequest([b'foo'])
        factory = ProxyClientFactory(b'GET', b'/foo', b'HTTP/1.0', {b'accept': b'text/html'}, '', request)
        factory.clientConnectionFailed(None, None)
        self.assertEqual(request.responseCode, 501)
        self.assertEqual(request.responseMessage, b'Gateway error')
        self.assertEqual(list(request.responseHeaders.getAllRawHeaders()), [(b'Content-Type', [b'text/html'])])
        self.assertEqual(b''.join(request.written), b'<H1>Could not connect</H1>')
        self.assertEqual(request.finished, 1)

    def test_buildProtocol(self):
        """
        L{ProxyClientFactory.buildProtocol} should produce a L{ProxyClient}
        with the same values of attributes (with updates on the headers).
        """
        factory = ProxyClientFactory(b'GET', b'/foo', b'HTTP/1.0', {b'accept': b'text/html'}, b'Some data', None)
        proto = factory.buildProtocol(None)
        self.assertIsInstance(proto, ProxyClient)
        self.assertEqual(proto.command, b'GET')
        self.assertEqual(proto.rest, b'/foo')
        self.assertEqual(proto.data, b'Some data')
        self.assertEqual(proto.headers, {b'accept': b'text/html', b'connection': b'close'})