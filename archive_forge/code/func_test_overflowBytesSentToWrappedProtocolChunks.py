from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_overflowBytesSentToWrappedProtocolChunks(self) -> None:
    """
        Test if header streaming passes extra data appropriately.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
    transport = StringTransportWithDisconnection()
    proto.makeConnection(transport)
    proto.dataReceived(self.IPV6HEADER[:18])
    proto.dataReceived(self.IPV6HEADER[18:] + b'HTTP/1.1 / GET')
    self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET')