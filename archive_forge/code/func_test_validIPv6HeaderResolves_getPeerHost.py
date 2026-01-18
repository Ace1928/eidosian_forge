from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_validIPv6HeaderResolves_getPeerHost(self) -> None:
    """
        Test if IPv6 headers result in the correct host and peer data.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.IPv4Address('TCP', '::1', 8080))
    transport = StringTransportWithDisconnection()
    proto.makeConnection(transport)
    proto.dataReceived(self.IPV6HEADER)
    self.assertEqual(proto.getPeer().host, '0:0:0:0:0:0:0:1')
    self.assertEqual(proto.getPeer().port, 8080)
    self.assertEqual(proto.wrappedProtocol.transport.getPeer().host, '0:0:0:0:0:0:0:1')
    self.assertEqual(proto.wrappedProtocol.transport.getPeer().port, 8080)
    self.assertEqual(proto.getHost().host, '0:0:0:0:0:0:0:1')
    self.assertEqual(proto.getHost().port, 8888)
    self.assertEqual(proto.wrappedProtocol.transport.getHost().host, '0:0:0:0:0:0:0:1')
    self.assertEqual(proto.wrappedProtocol.transport.getHost().port, 8888)