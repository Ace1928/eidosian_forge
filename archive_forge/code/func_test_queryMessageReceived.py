from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_queryMessageReceived(self):
    """
        L{DNSServerFactory.messageReceived} passes messages with an opcode of
        C{OP_QUERY} on to L{DNSServerFactory.handleQuery}.
        """
    self._messageReceivedTest('handleQuery', dns.Message(opCode=dns.OP_QUERY))