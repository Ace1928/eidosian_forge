from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_resolverOrder(self):
    """
        L{server.DNSServerFactory.resolver} contains an ordered list of
        authorities, caches and clients.
        """

    class DummyAuthority:
        pass

    class DummyCache:
        pass

    class DummyClient:
        pass
    self.assertEqual(server.DNSServerFactory(authorities=[DummyAuthority], caches=[DummyCache], clients=[DummyClient]).resolver.resolvers, [DummyAuthority, DummyCache, DummyClient])