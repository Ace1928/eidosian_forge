from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_sendReplyLoggingWithAnswers(self):
    """
        If L{server.DNSServerFactory.sendReply} logs a message for answers,
        authority, additional if the supplied a message has records in any of
        those sections.
        """
    self.patch(server.time, 'time', lambda: 86402)
    m = dns.Message()
    m.answers.append(dns.RRHeader(payload=dns.Record_A('127.0.0.1')))
    m.authority.append(dns.RRHeader(payload=dns.Record_A('127.0.0.1')))
    m.additional.append(dns.RRHeader(payload=dns.Record_A('127.0.0.1')))
    m.timeReceived = 86401
    f = server.DNSServerFactory(verbose=2)
    assertLogMessage(self, ['Answers are <A address=127.0.0.1 ttl=None>', 'Authority is <A address=127.0.0.1 ttl=None>', 'Additional is <A address=127.0.0.1 ttl=None>', 'Processed query in 1.000 seconds'], f.sendReply, protocol=NoopProtocol(), message=m, address=None)