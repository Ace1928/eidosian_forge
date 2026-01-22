from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
class URITestsForIPv6(URITests, unittest.TestCase):
    """
    Tests for L{twisted.web.client.URI} with IPv6 host addresses.

    IPv6 addresses must always be surrounded by square braces in URIs. No
    attempt is made to test without.
    """
    host = b'fe80::20c:29ff:fea4:c60'
    uriHost = b'[fe80::20c:29ff:fea4:c60]'

    def test_hostBracketIPv6AddressLiteral(self):
        """
        Brackets around IPv6 addresses are stripped in the host field. The host
        field is then exported with brackets in the output of
        L{client.URI.toBytes}.
        """
        uri = client.URI.fromBytes(b'http://[::1]:80/index.html')
        self.assertEqual(uri.host, b'::1')
        self.assertEqual(uri.netloc, b'[::1]:80')
        self.assertEqual(uri.toBytes(), b'http://[::1]:80/index.html')