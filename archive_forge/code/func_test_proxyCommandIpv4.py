from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_proxyCommandIpv4(self) -> None:
    """
        Test that proxy returns endpoint data for IPv4 connections.
        """
    header = _makeHeaderIPv4(verCom=b'!')
    info = _v2parser.V2Parser.parse(header)
    self.assertTrue(info.source)
    self.assertIsInstance(info.source, address.IPv4Address)
    self.assertTrue(info.destination)
    self.assertIsInstance(info.destination, address.IPv4Address)