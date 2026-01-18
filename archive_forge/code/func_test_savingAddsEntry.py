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
def test_savingAddsEntry(self):
    """
        L{KnownHostsFile.save} will write out a new file with any entries
        that have been added.
        """
    path = self.pathWithContent(sampleHashedLine + otherSamplePlaintextLine)
    knownHostsFile = KnownHostsFile.fromPath(path)
    newEntry = knownHostsFile.addHostKey(b'some.example.com', Key.fromString(thirdSampleKey))
    expectedContent = sampleHashedLine + otherSamplePlaintextLine + HashedEntry.MAGIC + b2a_base64(newEntry._hostSalt).strip() + b'|' + b2a_base64(newEntry._hostHash).strip() + b' ssh-rsa ' + thirdSampleEncodedKey + b'\n'
    self.assertEqual(3, expectedContent.count(b'\n'))
    knownHostsFile.save()
    self.assertEqual(expectedContent, path.getContent())