from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_responseFromMessageAuthoritativeMessage(self):
    """
        L{server.DNSServerFactory._responseFromMessage} marks the response
        message as authoritative if any of the answer records are authoritative.
        """
    factory = server.DNSServerFactory()
    response1 = factory._responseFromMessage(message=dns.Message(), answers=[dns.RRHeader(auth=True)])
    response2 = factory._responseFromMessage(message=dns.Message(), answers=[dns.RRHeader(auth=False)])
    self.assertEqual((True, False), (response1.auth, response2.auth))