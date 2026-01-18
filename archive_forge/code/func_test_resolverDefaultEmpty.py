from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_resolverDefaultEmpty(self):
    """
        L{server.DNSServerFactory.resolver} is an empty L{resolve.ResolverChain}
        by default.
        """
    self.assertEqual(server.DNSServerFactory().resolver.resolvers, [])