import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_ping(self):
    """
        L{IRCClient.ping}
        """
    self.protocol.ping('otheruser')
    self.assertTrue(self.transport.value().startswith(b'PRIVMSG otheruser :\x01PING'))
    self.transport.clear()
    self.protocol.ping('otheruser', 'are you there')
    self.assertEqual(self.transport.value(), b'PRIVMSG otheruser :\x01PING are you there\x01\r\n')
    self.transport.clear()
    self.protocol._pings = {}
    for pingNum in range(self.protocol._MAX_PINGRING + 3):
        self.protocol._pings['otheruser', str(pingNum)] = time.time() + pingNum
    self.assertEqual(len(self.protocol._pings), self.protocol._MAX_PINGRING + 3)
    self.protocol.ping('otheruser', 'I sent a lot of pings')
    self.assertEqual(len(self.protocol._pings), self.protocol._MAX_PINGRING)
    self.assertEqual(self.transport.value(), b'PRIVMSG otheruser :\x01PING I sent a lot of pings\x01\r\n')