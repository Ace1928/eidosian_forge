import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class QuotingTests(IRCTestCase):

    def test_lowquoteSanity(self):
        """
        Testing client-server level quote/dequote.
        """
        for s in stringSubjects:
            self.assertEqual(s, irc.lowDequote(irc.lowQuote(s)))

    def test_ctcpquoteSanity(self):
        """
        Testing CTCP message level quote/dequote.
        """
        for s in stringSubjects:
            self.assertEqual(s, irc.ctcpDequote(irc.ctcpQuote(s)))