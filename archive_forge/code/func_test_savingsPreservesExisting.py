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
def test_savingsPreservesExisting(self):
    """
        L{KnownHostsFile.save} will not overwrite existing entries in its save
        path, even if they were only added after the L{KnownHostsFile} instance
        was initialized.
        """
    path = self.pathWithContent(sampleHashedLine)
    knownHosts = KnownHostsFile.fromPath(path)
    with path.open('a') as hostsFileObj:
        hostsFileObj.write(otherSamplePlaintextLine)
    key = Key.fromString(thirdSampleKey)
    knownHosts.addHostKey(b'brandnew.example.com', key)
    knownHosts.save()
    knownHosts = KnownHostsFile.fromPath(path)
    self.assertEqual([True, True, True], [knownHosts.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)), knownHosts.hasHostKey(b'divmod.com', Key.fromString(otherSampleKey)), knownHosts.hasHostKey(b'brandnew.example.com', key)])