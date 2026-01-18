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
def test_notPresentKey(self):
    """
        L{KnownHostsFile.hasHostKey} returns C{False} when a key for the given
        hostname is not present.
        """
    hostsFile = self.loadSampleHostsFile()
    self.assertFalse(hostsFile.hasHostKey(b'non-existent.example.com', Key.fromString(sampleKey)))
    self.assertTrue(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)))
    self.assertFalse(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(ecdsaSampleKey)))