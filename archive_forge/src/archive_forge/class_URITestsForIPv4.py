from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
class URITestsForIPv4(URITests, unittest.TestCase):
    """
    Tests for L{twisted.web.client.URI} with IPv4 host addresses.
    """
    uriHost = host = b'192.168.1.67'