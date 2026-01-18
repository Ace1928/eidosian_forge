import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendUnicodeCommand(self):
    """
        Passing unicode parameters to L{IRC.sendCommand} encodes the parameters
        in UTF-8.
        """
    self.p.sendCommand('CMD', ('param¹', 'param²'))
    self.check(b'CMD param\xc2\xb9 param\xc2\xb2\r\n')