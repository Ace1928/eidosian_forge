import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendMessageInvalidCommand(self):
    """
        Passing an invalid string command to L{IRC.sendMessage} raises a
        C{ValueError}.
        """
    error = self.assertRaises(ValueError, self.p.sendMessage, ' ', 'param1', 'param2')
    self.assertEqual(str(error), "Somebody screwed up, 'cuz this doesn't look like a command to me:  ")