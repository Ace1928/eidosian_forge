from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_resolverType(self):
    """
        L{server.DNSServerFactory.resolver} is a L{resolve.ResolverChain}
        instance
        """
    self.assertIsInstance(server.DNSServerFactory().resolver, resolve.ResolverChain)