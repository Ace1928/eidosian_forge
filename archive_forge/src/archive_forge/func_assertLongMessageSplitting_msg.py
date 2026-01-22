import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def assertLongMessageSplitting_msg(self, message, expectedNumCommands, length=None):
    """
        Assert that messages sent by L{IRCClient.msg} are split into an
        expected number of commands and the original message is transmitted in
        its entirety over those commands.
        """
    responsePrefix = ':{}!{}@{} '.format(self.client.nickname, self.client.realname, self.client.hostname)
    self.client.msg('foo', message, length=length)
    privmsg = []
    self.patch(self.client, 'privmsg', lambda *a: privmsg.append(a))
    for line in self.client.lines:
        self.client.lineReceived(responsePrefix + line)
    self.assertEqual(len(privmsg), expectedNumCommands)
    receivedMessage = ''.join((message for user, target, message in privmsg))
    self.assertEqual(message, receivedMessage)