import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_singleLine(self):
    """
        A message containing no newlines is sent in a single command.
        """
    self.client.msg('foo', 'bar')
    self.assertEqual(self.client.lines, ['PRIVMSG foo :bar'])