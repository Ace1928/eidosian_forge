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
class PlainEntryTests(EntryTestsMixin, TestCase):
    """
    Test cases for L{PlainEntry}.
    """
    plaintextLine = samplePlaintextLine
    hostIPLine = sampleHostIPLine

    def setUp(self):
        """
        Set 'entry' to a sample plain-text entry with sampleKey as its key.
        """
        self.entry = PlainEntry.fromString(self.plaintextLine)

    def test_matchesHostIP(self):
        """
        A "hostname,ip" formatted line will match both the host and the IP.
        """
        self.entry = PlainEntry.fromString(self.hostIPLine)
        self.assertTrue(self.entry.matchesHost(b'198.49.126.131'))
        self.test_matchesHost()

    def test_toString(self):
        """
        L{PlainEntry.toString} generates the serialized OpenSSL format string
        for the entry, sans newline.
        """
        self.assertEqual(self.entry.toString(), self.plaintextLine.rstrip(b'\n'))
        multiHostEntry = PlainEntry.fromString(self.hostIPLine)
        self.assertEqual(multiHostEntry.toString(), self.hostIPLine.rstrip(b'\n'))