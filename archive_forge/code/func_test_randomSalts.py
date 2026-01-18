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
def test_randomSalts(self):
    """
        L{KnownHostsFile.addHostKey} generates a random salt for each new key,
        so subsequent salts will be different.
        """
    hostsFile = self.loadSampleHostsFile()
    aKey = Key.fromString(thirdSampleKey)
    self.assertNotEqual(hostsFile.addHostKey(b'somewhere.example.com', aKey)._hostSalt, hostsFile.addHostKey(b'somewhere-else.example.com', aKey)._hostSalt)