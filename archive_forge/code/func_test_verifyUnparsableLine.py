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
def test_verifyUnparsableLine(self):
    """
        Loading a L{KnownHostsFile} from a path that contains an unparseable
        line will be represented as an L{UnparsedEntry} instance.
        """
    hostsFile = self.loadSampleHostsFile(b'This is just unparseable.\n')
    entries = list(hostsFile.iterentries())
    self.assertIsInstance(entries[0], UnparsedEntry)
    self.assertEqual(entries[0].toString(), b'This is just unparseable.')
    self.assertEqual(1, len(entries))