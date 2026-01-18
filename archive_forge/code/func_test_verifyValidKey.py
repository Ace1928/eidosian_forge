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
def test_verifyValidKey(self):
    """
        Verifying a valid key should return a L{Deferred} which fires with
        True.
        """
    hostsFile = self.loadSampleHostsFile()
    hostsFile.addHostKey(b'1.2.3.4', Key.fromString(sampleKey))
    ui = FakeUI()
    d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'1.2.3.4', Key.fromString(sampleKey))
    l = []
    d.addCallback(l.append)
    self.assertEqual(l, [True])