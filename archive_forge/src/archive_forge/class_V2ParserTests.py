from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
class V2ParserTests(unittest.TestCase):
    """
    Test L{twisted.protocols.haproxy.V2Parser} behaviour.
    """

    def test_happyPathIPv4(self) -> None:
        """
        Test if a well formed IPv4 header is parsed without error.
        """
        header = _makeHeaderIPv4()
        self.assertTrue(_v2parser.V2Parser.parse(header))

    def test_happyPathIPv6(self) -> None:
        """
        Test if a well formed IPv6 header is parsed without error.
        """
        header = _makeHeaderIPv6()
        self.assertTrue(_v2parser.V2Parser.parse(header))

    def test_happyPathUnix(self) -> None:
        """
        Test if a well formed UNIX header is parsed without error.
        """
        header = _makeHeaderUnix()
        self.assertTrue(_v2parser.V2Parser.parse(header))

    def test_invalidSignature(self) -> None:
        """
        Test if an invalid signature block raises InvalidProxyError.
        """
        header = _makeHeaderIPv4(sig=b'\x00' * 12)
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidVersion(self) -> None:
        """
        Test if an invalid version raises InvalidProxyError.
        """
        header = _makeHeaderIPv4(verCom=b'\x11')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidCommand(self) -> None:
        """
        Test if an invalid command raises InvalidProxyError.
        """
        header = _makeHeaderIPv4(verCom=b'#')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidFamily(self) -> None:
        """
        Test if an invalid family raises InvalidProxyError.
        """
        header = _makeHeaderIPv4(famProto=b'@')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_invalidProto(self) -> None:
        """
        Test if an invalid protocol raises InvalidProxyError.
        """
        header = _makeHeaderIPv4(famProto=b'$')
        self.assertRaises(InvalidProxyHeader, _v2parser.V2Parser.parse, header)

    def test_localCommandIpv4(self) -> None:
        """
        Test that local does not return endpoint data for IPv4 connections.
        """
        header = _makeHeaderIPv4(verCom=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_localCommandIpv6(self) -> None:
        """
        Test that local does not return endpoint data for IPv6 connections.
        """
        header = _makeHeaderIPv6(verCom=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_localCommandUnix(self) -> None:
        """
        Test that local does not return endpoint data for UNIX connections.
        """
        header = _makeHeaderUnix(verCom=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

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

    def test_proxyCommandUnix(self) -> None:
        """
        Test that proxy returns endpoint data for UNIX connections.
        """
        header = _makeHeaderUnix(verCom=b'!')
        info = _v2parser.V2Parser.parse(header)
        self.assertTrue(info.source)
        self.assertIsInstance(info.source, address.UNIXAddress)
        self.assertTrue(info.destination)
        self.assertIsInstance(info.destination, address.UNIXAddress)

    def test_unspecFamilyIpv4(self) -> None:
        """
        Test that UNSPEC does not return endpoint data for IPv4 connections.
        """
        header = _makeHeaderIPv4(famProto=b'\x01')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecFamilyIpv6(self) -> None:
        """
        Test that UNSPEC does not return endpoint data for IPv6 connections.
        """
        header = _makeHeaderIPv6(famProto=b'\x01')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecFamilyUnix(self) -> None:
        """
        Test that UNSPEC does not return endpoint data for UNIX connections.
        """
        header = _makeHeaderUnix(famProto=b'\x01')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecProtoIpv4(self) -> None:
        """
        Test that UNSPEC does not return endpoint data for IPv4 connections.
        """
        header = _makeHeaderIPv4(famProto=b'\x10')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecProtoIpv6(self) -> None:
        """
        Test that UNSPEC does not return endpoint data for IPv6 connections.
        """
        header = _makeHeaderIPv6(famProto=b' ')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_unspecProtoUnix(self) -> None:
        """
        Test that UNSPEC does not return endpoint data for UNIX connections.
        """
        header = _makeHeaderUnix(famProto=b'0')
        info = _v2parser.V2Parser.parse(header)
        self.assertFalse(info.source)
        self.assertFalse(info.destination)

    def test_overflowIpv4(self) -> None:
        """
        Test that overflow bits are preserved during feed parsing for IPv4.
        """
        testValue = b'TEST DATA\r\n\r\nTEST DATA'
        header = _makeHeaderIPv4() + testValue
        parser = _v2parser.V2Parser()
        info, overflow = parser.feed(header)
        self.assertTrue(info)
        self.assertEqual(overflow, testValue)

    def test_overflowIpv6(self) -> None:
        """
        Test that overflow bits are preserved during feed parsing for IPv6.
        """
        testValue = b'TEST DATA\r\n\r\nTEST DATA'
        header = _makeHeaderIPv6() + testValue
        parser = _v2parser.V2Parser()
        info, overflow = parser.feed(header)
        self.assertTrue(info)
        self.assertEqual(overflow, testValue)

    def test_overflowUnix(self) -> None:
        """
        Test that overflow bits are preserved during feed parsing for Unix.
        """
        testValue = b'TEST DATA\r\n\r\nTEST DATA'
        header = _makeHeaderUnix() + testValue
        parser = _v2parser.V2Parser()
        info, overflow = parser.feed(header)
        self.assertTrue(info)
        self.assertEqual(overflow, testValue)

    def test_segmentTooSmall(self) -> None:
        """
        Test that an initial payload of less than 16 bytes fails.
        """
        testValue = b'NEEDMOREDATA'
        parser = _v2parser.V2Parser()
        self.assertRaises(InvalidProxyHeader, parser.feed, testValue)