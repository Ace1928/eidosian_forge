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
def test_verifyUnparsableEncryptionMarker(self):
    """
        Loading a L{KnownHostsFile} from a path containing an unparseable line
        that starts with an encryption marker will be represented as an
        L{UnparsedEntry} instance.
        """
    hostsFile = self.loadSampleHostsFile(b'|1|This is unparseable.\n')
    entries = list(hostsFile.iterentries())
    self.assertIsInstance(entries[0], UnparsedEntry)
    self.assertEqual(entries[0].toString(), b'|1|This is unparseable.')
    self.assertEqual(1, len(entries))