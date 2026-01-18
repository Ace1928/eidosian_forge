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
def test_notInKnownHosts(self):
    """
        L{default.isInKnownHosts} should return C{0} when a host with a key
        is not in the known hosts file.
        """
    r = default.isInKnownHosts('not.there', b'irrelevant', {'known-hosts': FilePath(self.hostsOption).path})
    self.assertEqual(0, r)