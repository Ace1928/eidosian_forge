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
def test_pickServerSamePriorities(self):
    """
        Two records with equal priorities compare on weight (ascending).
        """
    record1 = dns.Record_SRV(10, 10, 5222, 'host1.example.org')
    record2 = dns.Record_SRV(10, 20, 5222, 'host2.example.org')
    self.connector.orderedServers = [record2, record1]
    self.connector.servers = []
    self.patch(random, 'randint', self._randint)
    self.randIntResults = [0, 0]
    self.assertEqual(('host1.example.org', 5222), self.connector.pickServer())
    self.assertEqual(('host2.example.org', 5222), self.connector.pickServer())