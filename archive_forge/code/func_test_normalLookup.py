import time
from zope.interface.verify import verifyClass
from twisted.internet import interfaces, task
from twisted.names import cache, dns
from twisted.trial import unittest
def test_normalLookup(self):
    """
        When a cache lookup finds a cached entry from 1 second ago, it is
        returned with a TTL of original TTL minus the elapsed 1 second.
        """
    r = ([dns.RRHeader(b'example.com', dns.A, dns.IN, 60, dns.Record_A('127.0.0.1', 60))], [dns.RRHeader(b'example.com', dns.A, dns.IN, 50, dns.Record_A('127.0.0.1', 50))], [dns.RRHeader(b'example.com', dns.A, dns.IN, 40, dns.Record_A('127.0.0.1', 40))])
    clock = task.Clock()
    c = cache.CacheResolver(reactor=clock)
    c.cacheResult(dns.Query(name=b'example.com', type=dns.A, cls=dns.IN), r)
    clock.advance(1)

    def cbLookup(result):
        self.assertEqual(result[0][0].ttl, 59)
        self.assertEqual(result[1][0].ttl, 49)
        self.assertEqual(result[2][0].ttl, 39)
        self.assertEqual(result[0][0].name.name, b'example.com')
    return c.lookupAddress(b'example.com').addCallback(cbLookup)