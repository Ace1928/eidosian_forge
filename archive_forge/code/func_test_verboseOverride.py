from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_verboseOverride(self):
    """
        L{server.DNSServerFactory.__init__} accepts a C{verbose} argument which
        overrides L{server.DNSServerFactory.verbose}.
        """
    self.assertTrue(server.DNSServerFactory(verbose=True).verbose)