import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommandInvalidCommand(self):
    """
        Passing an invalid string command to L{IRC.sendCommand} raises a
        C{ValueError}.
        """
    error = self.assertRaises(ValueError, self.p.sendCommand, ' ', ('param1', 'param2'))
    self.assertEqual(error.args[0], 'Invalid command: " "')