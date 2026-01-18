from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_responseFromMessageTimeReceived(self):
    """
        L{server.DNSServerFactory._responseFromMessage} generates a response
        message whose C{timeReceived} attribute has the same value as that found
        on the request.
        """
    factory = server.DNSServerFactory()
    request = dns.Message()
    request.timeReceived = 1234
    response = factory._responseFromMessage(message=request)
    self.assertEqual(request.timeReceived, response.timeReceived)