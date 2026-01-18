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
def test_verifyNonPresentKey_No(self):
    """
        Verifying a key where neither the hostname nor the IP are present
        should result in the UI being prompted with a message explaining as
        much.  If the UI says no, the Deferred should fail with
        UserRejectedKey.
        """
    ui, l, knownHostsFile = self.verifyNonPresentKey()
    ui.promptDeferred.callback(False)
    l[0].trap(UserRejectedKey)