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
def test_unsavedEntryHasKeyMismatch(self):
    """
        L{KnownHostsFile.hasHostKey} raises L{HostKeyChanged} if the host key is
        present in memory (but not yet saved), but different from the expected
        one.  The resulting exception has a C{offendingEntry} indicating the
        given entry, but no filename or line number information (reflecting the
        fact that the entry exists only in memory).
        """
    hostsFile = KnownHostsFile(FilePath(self.mktemp()))
    entry = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
    exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.example.com', Key.fromString(thirdSampleKey))
    self.assertEqual(exception.offendingEntry, entry)
    self.assertIsNone(exception.lineno)
    self.assertIsNone(exception.path)