from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
class ReverseProxyRequestTests(TestCase):
    """
    Tests for L{ReverseProxyRequest}.
    """

    def test_process(self):
        """
        L{ReverseProxyRequest.process} should create a connection to its
        factory host/port, using a L{ProxyClientFactory} instantiated with the
        correct parameters, and particularly set the B{host} header to the
        factory host.
        """
        transport = StringTransportWithDisconnection()
        channel = DummyChannel(transport)
        reactor = MemoryReactor()
        request = ReverseProxyRequest(channel, False, reactor)
        request.factory = DummyFactory('example.com', 1234)
        request.gotLength(0)
        request.requestReceived(b'GET', b'/foo/bar', b'HTTP/1.0')
        self.assertEqual(len(reactor.tcpClients), 1)
        self.assertEqual(reactor.tcpClients[0][0], 'example.com')
        self.assertEqual(reactor.tcpClients[0][1], 1234)
        factory = reactor.tcpClients[0][2]
        self.assertIsInstance(factory, ProxyClientFactory)
        self.assertEqual(factory.headers, {b'host': b'example.com'})