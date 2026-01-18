from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_verboseLogQuiet(self):
    """
        L{server.DNSServerFactory._verboseLog} does not log messages unless
        C{verbose > 0}.
        """
    f = server.DNSServerFactory()
    assertLogMessage(self, [], f._verboseLog, 'Foo Bar')