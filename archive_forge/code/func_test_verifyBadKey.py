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
def test_verifyBadKey(self):
    """
        L{default.verifyHostKey} should return a L{Deferred} which fails with
        L{HostKeyChanged} if the host key is incorrect.
        """
    d = default.verifyHostKey(self.fakeTransport, b'4.3.2.1', otherSampleKey, 'Again, not required.')
    return self.assertFailure(d, HostKeyChanged)