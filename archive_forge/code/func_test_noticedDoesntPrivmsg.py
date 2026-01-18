import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_noticedDoesntPrivmsg(self):
    """
        The default implementation of L{IRCClient.noticed} doesn't invoke
        C{privmsg()}
        """

    def privmsg(user, channel, message):
        self.fail('privmsg() should not have been called')
    self.protocol.privmsg = privmsg
    self.protocol.irc_NOTICE('spam', ['#greasyspooncafe', "I don't want any spam!"])