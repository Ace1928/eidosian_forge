from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_proxyCommandIpv6(self) -> None:
    """
        Test that proxy returns endpoint data for IPv6 connections.
        """
    header = _makeHeaderIPv6(verCom=b'!')
    info = _v2parser.V2Parser.parse(header)
    self.assertTrue(info.source)
    self.assertIsInstance(info.source, address.IPv6Address)
    self.assertTrue(info.destination)
    self.assertIsInstance(info.destination, address.IPv6Address)