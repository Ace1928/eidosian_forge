from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_responseFromMessageRecursionAvailable(self):
    """
        L{server.DNSServerFactory._responseFromMessage} generates a response
        message whose C{recAV} attribute is L{True} if
        L{server.DNSServerFactory.canRecurse} is L{True}.
        """
    factory = server.DNSServerFactory()
    factory.canRecurse = True
    response1 = factory._responseFromMessage(message=dns.Message(recAv=False))
    factory.canRecurse = False
    response2 = factory._responseFromMessage(message=dns.Message(recAv=True))
    self.assertEqual((True, False), (response1.recAv, response2.recAv))