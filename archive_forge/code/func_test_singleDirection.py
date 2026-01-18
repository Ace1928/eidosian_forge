import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_singleDirection(self):
    """
        Parsing a single-direction mode setting with multiple modes and no
        parameters, results in all modes falling into the same direction group.
        """
    added, removed = irc.parseModes('+stn', [])
    self.assertEqual(added, [('s', None), ('t', None), ('n', None)])
    self.assertEqual(removed, [])
    added, removed = irc.parseModes('-nt', [])
    self.assertEqual(added, [])
    self.assertEqual(removed, [('n', None), ('t', None)])