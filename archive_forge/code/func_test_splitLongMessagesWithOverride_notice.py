import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_splitLongMessagesWithOverride_notice(self):
    """
        The maximum message length can be specified to L{IRCClient.notice},
        messages longer than this are split into multiple commands with the
        intent of delivering long messages without losing data due to message
        truncation when the server relays them.
        """
    message = 'o' * (irc.MAX_COMMAND_LENGTH - 2)
    self.assertLongMessageSplitting_notice(message, 3, length=irc.MAX_COMMAND_LENGTH // 2)