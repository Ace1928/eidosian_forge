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
def test_reentrantTCPQueryErrbackOnConnectionFailure(self):
    """
        An errback on the deferred returned by
        L{client.Resolver.queryTCP} may trigger another TCP query.
        """
    reactor = proto_helpers.MemoryReactor()
    resolver = client.Resolver(servers=[('127.0.0.1', 10053)], reactor=reactor)
    q = dns.Query('example.com')
    d = resolver.queryTCP(q)

    def reissue(e):
        e.trap(ConnectionRefusedError)
        return resolver.queryTCP(q)
    d.addErrback(reissue)
    self.assertEqual(len(reactor.tcpClients), 1)
    self.assertEqual(len(reactor.connectors), 1)
    host, port, factory, timeout, bindAddress = reactor.tcpClients[0]
    f1 = failure.Failure(ConnectionRefusedError())
    factory.clientConnectionFailed(reactor.connectors[0], f1)
    self.assertEqual(len(reactor.tcpClients), 2)
    self.assertEqual(len(reactor.connectors), 2)
    self.assertNoResult(d)
    f2 = failure.Failure(ConnectionRefusedError())
    factory.clientConnectionFailed(reactor.connectors[1], f2)
    f = self.failureResultOf(d, ConnectionRefusedError)
    self.assertIs(f, f2)