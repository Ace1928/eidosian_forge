from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_sendReplyWithoutAddress(self):
    """
        If L{server.DNSServerFactory.sendReply} is supplied with a protocol but
        no address tuple it will supply only a message to
        C{protocol.writeMessage}.
        """
    m = dns.Message()
    f = server.DNSServerFactory()
    e = self.assertRaises(RaisingProtocol.WriteMessageArguments, f.sendReply, protocol=RaisingProtocol(), message=m, address=None)
    args, kwargs = e.args
    self.assertEqual(args, (m,))
    self.assertEqual(kwargs, {})