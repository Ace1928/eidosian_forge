from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_validUNIXHeaderResolves_getPeerHost(self) -> None:
    """
        Test if UNIX headers result in the correct host and peer data.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.UNIXAddress(b'/home/test/sockets/server.sock'))
    transport = StringTransportWithDisconnection()
    proto.makeConnection(transport)
    proto.dataReceived(self.UNIXHEADER)
    self.assertEqual(proto.getPeer().name, b'/home/tests/mysockets/sock')
    self.assertEqual(proto.wrappedProtocol.transport.getPeer().name, b'/home/tests/mysockets/sock')
    self.assertEqual(proto.getHost().name, b'/home/tests/mysockets/sock')
    self.assertEqual(proto.wrappedProtocol.transport.getHost().name, b'/home/tests/mysockets/sock')