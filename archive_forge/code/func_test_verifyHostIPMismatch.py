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
def test_verifyHostIPMismatch(self):
    """
        Verifying a key where the host is present (and correct), but the IP is
        present and different, should result the deferred firing in a
        HostKeyChanged failure.
        """
    hostsFile = self.loadSampleHostsFile()
    wrongKey = Key.fromString(thirdSampleKey)
    ui = FakeUI()
    d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'4.3.2.1', wrongKey)
    return self.assertFailure(d, HostKeyChanged)