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
def test_verifyQuestion(self):
    """
        L{default.verifyHostKey} should return a L{Default} which fires with
        C{0} when passed an unknown host that the user refuses to acknowledge.
        """
    self.fakeTransport.factory.options['host'] = b'fake.example.com'
    self.fakeFile.inlines.append(b'no')
    d = default.verifyHostKey(self.fakeTransport, b'9.8.7.6', otherSampleKey, b'No fingerprint!')
    self.assertEqual([b"The authenticity of host 'fake.example.com (9.8.7.6)' can't be established.\nRSA key fingerprint is SHA256:vD0YydsNIUYJa7yLZl3tIL8h0vZvQ8G+HPG7JLmQV0s=.\nAre you sure you want to continue connecting (yes/no)? "], self.fakeFile.outchunks)
    return self.assertFailure(d, UserRejectedKey)