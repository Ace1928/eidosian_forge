import random
from zope.interface.verify import verifyObject
from twisted.internet import defer, protocol
from twisted.internet.error import DNSLookupError, ServiceNameUnknownError
from twisted.internet.interfaces import IConnector
from twisted.internet.testing import MemoryReactor
from twisted.names import client, dns, srvconnect
from twisted.names.common import ResolverBase
from twisted.names.error import DNSNameError
from twisted.trial import unittest
def test_unicodeDomain(self):
    """
        L{srvconnect.SRVConnector} automatically encodes unicode domain using
        C{idna} encoding.
        """
    self.connector = srvconnect.SRVConnector(self.reactor, 'xmpp-client', 'échec.example.org', self.factory)
    self.assertEqual(b'xn--chec-9oa.example.org', self.connector.domain)