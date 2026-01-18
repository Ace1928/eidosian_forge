import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_whois(self):
    """
        L{IRCClient.whois} sends a WHOIS message.
        """
    self.protocol.whois('alice')
    self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), ['WHOIS alice', ''])