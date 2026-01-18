import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_splitValidatesLength(self):
    """
        L{twisted.words.protocols.irc.split} raises C{ValueError} if given a
        length less than or equal to C{0}.
        """
    self.assertRaises(ValueError, irc.split, 'foo', 0)
    self.assertRaises(ValueError, irc.split, 'foo', -1)