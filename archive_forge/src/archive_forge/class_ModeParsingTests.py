import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class ModeParsingTests(IRCTestCase):
    """
    Tests for L{twisted.words.protocols.irc.parseModes}.
    """
    paramModes = ('klb', 'b')

    def test_emptyModes(self):
        """
        Parsing an empty mode string raises L{irc.IRCBadModes}.
        """
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '', [])

    def test_emptyModeSequence(self):
        """
        Parsing a mode string that contains an empty sequence (either a C{+} or
        C{-} followed directly by another C{+} or C{-}, or not followed by
        anything at all) raises L{irc.IRCBadModes}.
        """
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '++k', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '-+k', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '-', [])

    def test_malformedModes(self):
        """
        Parsing a mode string that does not start with C{+} or C{-} raises
        L{irc.IRCBadModes}.
        """
        self.assertRaises(irc.IRCBadModes, irc.parseModes, 'foo', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '%', [])

    def test_nullModes(self):
        """
        Parsing a mode string that contains no mode characters raises
        L{irc.IRCBadModes}.
        """
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '-', [])

    def test_singleMode(self):
        """
        Parsing a single mode setting with no parameters results in that mode,
        with no parameters, in the "added" direction and no modes in the
        "removed" direction.
        """
        added, removed = irc.parseModes('+s', [])
        self.assertEqual(added, [('s', None)])
        self.assertEqual(removed, [])
        added, removed = irc.parseModes('-s', [])
        self.assertEqual(added, [])
        self.assertEqual(removed, [('s', None)])

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

    def test_multiDirection(self):
        """
        Parsing a multi-direction mode setting with no parameters.
        """
        added, removed = irc.parseModes('+s-n+ti', [])
        self.assertEqual(added, [('s', None), ('t', None), ('i', None)])
        self.assertEqual(removed, [('n', None)])

    def test_consecutiveDirection(self):
        """
        Parsing a multi-direction mode setting containing two consecutive mode
        sequences with the same direction results in the same result as if
        there were only one mode sequence in the same direction.
        """
        added, removed = irc.parseModes('+sn+ti', [])
        self.assertEqual(added, [('s', None), ('n', None), ('t', None), ('i', None)])
        self.assertEqual(removed, [])

    def test_mismatchedParams(self):
        """
        If the number of mode parameters does not match the number of modes
        expecting parameters, L{irc.IRCBadModes} is raised.
        """
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+k', [], self.paramModes)
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+kl', ['foo', '10', 'lulz_extra_param'], self.paramModes)

    def test_parameters(self):
        """
        Modes which require parameters are parsed and paired with their relevant
        parameter, modes which do not require parameters do not consume any of
        the parameters.
        """
        added, removed = irc.parseModes('+klbb', ['somekey', '42', 'nick!user@host', 'other!*@*'], self.paramModes)
        self.assertEqual(added, [('k', 'somekey'), ('l', '42'), ('b', 'nick!user@host'), ('b', 'other!*@*')])
        self.assertEqual(removed, [])
        added, removed = irc.parseModes('-klbb', ['nick!user@host', 'other!*@*'], self.paramModes)
        self.assertEqual(added, [])
        self.assertEqual(removed, [('k', None), ('l', None), ('b', 'nick!user@host'), ('b', 'other!*@*')])
        added, removed = irc.parseModes('+knbb', ['somekey', 'nick!user@host', 'other!*@*'], self.paramModes)
        self.assertEqual(added, [('k', 'somekey'), ('n', None), ('b', 'nick!user@host'), ('b', 'other!*@*')])
        self.assertEqual(removed, [])