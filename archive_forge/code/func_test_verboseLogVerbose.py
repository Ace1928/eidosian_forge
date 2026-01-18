from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_verboseLogVerbose(self):
    """
        L{server.DNSServerFactory._verboseLog} logs a message if C{verbose > 0}.
        """
    f = server.DNSServerFactory(verbose=1)
    assertLogMessage(self, ['Foo Bar'], f._verboseLog, 'Foo Bar')