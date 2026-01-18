import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_overrideAlterCollidedNick(self):
    """
        L{IRCClient.alterCollidedNick} determines how a nickname is altered upon
        collision while a user is trying to change to that nickname.
        """
    nick = 'foo'
    self.protocol.alterCollidedNick = lambda nick: nick + '***'
    self.protocol.register(nick)
    self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
    lastLine = self.getLastLine(self.transport)
    self.assertEqual(lastLine, 'NICK {}'.format(nick + '***'))