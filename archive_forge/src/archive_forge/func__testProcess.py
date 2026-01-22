from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def _testProcess(self, uri, expectedURI, method=b'GET', data=b''):
    """
        Build a request pointing at C{uri}, and check that a proxied request
        is created, pointing a C{expectedURI}.
        """
    transport = StringTransportWithDisconnection()
    channel = DummyChannel(transport)
    reactor = MemoryReactor()
    request = ProxyRequest(channel, False, reactor)
    request.gotLength(len(data))
    request.handleContentChunk(data)
    request.requestReceived(method, b'http://example.com' + uri, b'HTTP/1.0')
    self.assertEqual(len(reactor.tcpClients), 1)
    self.assertEqual(reactor.tcpClients[0][0], 'example.com')
    self.assertEqual(reactor.tcpClients[0][1], 80)
    factory = reactor.tcpClients[0][2]
    self.assertIsInstance(factory, ProxyClientFactory)
    self.assertEqual(factory.command, method)
    self.assertEqual(factory.version, b'HTTP/1.0')
    self.assertEqual(factory.headers, {b'host': b'example.com'})
    self.assertEqual(factory.data, data)
    self.assertEqual(factory.rest, expectedURI)
    self.assertEqual(factory.father, request)