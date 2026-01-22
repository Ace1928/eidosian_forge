from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
class RaisingDNSServerFactory(server.DNSServerFactory):

    def allowQuery(self, *args, **kwargs):
        return False

    def sendReply(self, *args, **kwargs):
        raise SendReplyException(args, kwargs)