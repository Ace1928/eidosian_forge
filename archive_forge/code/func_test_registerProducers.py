from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_registerProducers(self):
    """
        The proxy client registers itself as a producer of the proxy server and
        vice versa.
        """
    addr = address.IPv4Address('TCP', '127.0.0.1', 0)
    server = portforward.ProxyFactory('127.0.0.1', 0).buildProtocol(addr)
    reactor = proto_helpers.MemoryReactor()
    server.reactor = reactor
    serverTransport = proto_helpers.StringTransport()
    server.makeConnection(serverTransport)
    self.assertEqual(len(reactor.tcpClients), 1)
    host, port, clientFactory, timeout, _ = reactor.tcpClients[0]
    self.assertIsInstance(clientFactory, portforward.ProxyClientFactory)
    client = clientFactory.buildProtocol(addr)
    clientTransport = proto_helpers.StringTransport()
    client.makeConnection(clientTransport)
    self.assertIs(clientTransport.producer, serverTransport)
    self.assertIs(serverTransport.producer, clientTransport)
    self.assertTrue(clientTransport.streaming)
    self.assertTrue(serverTransport.streaming)