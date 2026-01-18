import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_KICKLEN(self):
    """
        The KICKLEN support parameter is parsed into an integer value
        indicating the maximum length of a kick message a client may use.
        """
    self._testIntOrDefaultFeature('KICKLEN')