from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
class NoResponseDNSServerFactory(server.DNSServerFactory):
    """
    A L{server.DNSServerFactory} subclass which does not attempt to reply to any
    received messages.

    Used for testing logged messages in C{messageReceived} without having to
    fake or patch the preceding code which attempts to deliver a response
    message.
    """

    def allowQuery(self, message, protocol, address):
        """
        Deny all queries.

        @param message: See L{server.DNSServerFactory.allowQuery}
        @param protocol: See L{server.DNSServerFactory.allowQuery}
        @param address: See L{server.DNSServerFactory.allowQuery}

        @return: L{False}
        @rtype: L{bool}
        """
        return False

    def sendReply(self, protocol, message, address):
        """
        A noop send reply.

        @param protocol: See L{server.DNSServerFactory.sendReply}
        @param message: See L{server.DNSServerFactory.sendReply}
        @param address: See L{server.DNSServerFactory.sendReply}
        """