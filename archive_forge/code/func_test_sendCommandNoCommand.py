import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommandNoCommand(self):
    """
        Passing L{None} as the command to L{IRC.sendCommand} raises a
        C{ValueError}.
        """
    error = self.assertRaises(ValueError, self.p.sendCommand, None, ('param1', 'param2'))
    self.assertEqual(error.args[0], 'IRC message requires a command.')