import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
class UnparsedEntryTests(TestCase, EntryTestsMixin):
    """
    Tests for L{UnparsedEntry}
    """

    def setUp(self):
        """
        Set up the 'entry' to be an unparsed entry for some random text.
        """
        self.entry = UnparsedEntry(b'    This is a bogus entry.  \n')

    def test_fromString(self):
        """
        Creating an L{UnparsedEntry} should simply record the string it was
        passed.
        """
        self.assertEqual(b'    This is a bogus entry.  \n', self.entry._string)

    def test_matchesHost(self):
        """
        An unparsed entry can't match any hosts.
        """
        self.assertFalse(self.entry.matchesHost(b'www.twistedmatrix.com'))

    def test_matchesKey(self):
        """
        An unparsed entry can't match any keys.
        """
        self.assertFalse(self.entry.matchesKey(Key.fromString(sampleKey)))

    def test_toString(self):
        """
        L{UnparsedEntry.toString} returns its input string, sans trailing
        newline.
        """
        self.assertEqual(b'    This is a bogus entry.  ', self.entry.toString())