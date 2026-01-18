from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_parseDefaultPort(self):
    """
        L{client.URI.fromBytes} by default assumes port 80 for the I{http}
        scheme and 443 for the I{https} scheme.
        """
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST'))
    self.assertEqual(80, uri.port)
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST:'))
    self.assertEqual(80, uri.port)
    uri = client.URI.fromBytes(self.makeURIString(b'https://HOST'))
    self.assertEqual(443, uri.port)