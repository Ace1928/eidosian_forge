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
def test_protocolShutDownAfterTimeout(self):
    """
        The L{DNSDatagramProtocol} created when an interim timeout occurs is
        also disconnected from its transport after the Deferred returned by its
        query method completes.
        """
    resolver = client.Resolver(servers=[('example.com', 53)])
    protocols = []
    result = defer.Deferred()
    results = [defer.fail(failure.Failure(DNSQueryTimeoutError(None))), result]

    class FakeProtocol:

        def __init__(self):
            self.transport = StubPort()

        def query(self, address, query, timeout=10, id=None):
            protocols.append(self)
            return results.pop(0)
    resolver._connectedProtocol = FakeProtocol
    resolver.query(dns.Query(b'foo.example.com'))
    self.assertFalse(protocols[1].transport.disconnected)
    result.callback(dns.Message())
    self.assertTrue(protocols[1].transport.disconnected)