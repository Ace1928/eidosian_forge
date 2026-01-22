import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _truncateLine(self, line):
    """
        Truncate an IRC line to the maximum allowed length.
        """
    return line[:irc.MAX_COMMAND_LENGTH - len(self.delimiter)]