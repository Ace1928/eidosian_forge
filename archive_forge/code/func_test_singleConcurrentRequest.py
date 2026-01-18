import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_singleConcurrentRequest(self):
    """
        L{client.Resolver.query} only issues one request at a time per query.
        Subsequent requests made before responses to prior ones are received
        are queued and given the same response as is given to the first one.
        """
    protocol = StubDNSDatagramProtocol()
    resolver = client.Resolver(servers=[('example.com', 53)])
    resolver._connectedProtocol = lambda: protocol
    queries = protocol.queries
    query = dns.Query(b'foo.example.com', dns.A, dns.IN)
    firstResult = resolver.query(query)
    self.assertEqual(len(queries), 1)
    secondResult = resolver.query(query)
    self.assertEqual(len(queries), 1)
    answer = object()
    response = dns.Message()
    response.answers.append(answer)
    queries.pop()[-1].callback(response)
    d = defer.gatherResults([firstResult, secondResult])

    def cbFinished(responses):
        firstResponse, secondResponse = responses
        self.assertEqual(firstResponse, ([answer], [], []))
        self.assertEqual(secondResponse, ([answer], [], []))
    d.addCallback(cbFinished)
    return d