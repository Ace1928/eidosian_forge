import os.path
from errno import ENOSYS
from struct import pack
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
import hamcrest
from twisted.conch.error import ConchError, HostKeyChanged, UserRejectedKey
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.portal import Portal
from twisted.internet.address import IPv4Address
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import (
from twisted.internet.interfaces import IAddress, IStreamClientEndpoint
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import (
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
from twisted.test.iosim import FakeTransport, connect
class ExistingConnectionHelperTests(TestCase):
    """
    Tests for L{_ExistingConnectionHelper}.
    """

    def test_interface(self):
        """
        L{_ExistingConnectionHelper} implements L{_ISSHConnectionCreator}.
        """
        self.assertTrue(verifyClass(_ISSHConnectionCreator, _ExistingConnectionHelper))

    def test_secureConnection(self):
        """
        L{_ExistingConnectionHelper.secureConnection} returns a L{Deferred}
        which fires with whatever object was fed to
        L{_ExistingConnectionHelper.__init__}.
        """
        result = object()
        helper = _ExistingConnectionHelper(result)
        self.assertIs(result, self.successResultOf(helper.secureConnection()))

    def test_cleanupConnectionNotImmediately(self):
        """
        L{_ExistingConnectionHelper.cleanupConnection} does nothing to the
        existing connection if called with C{immediate} set to C{False}.
        """
        helper = _ExistingConnectionHelper(object())
        helper.cleanupConnection(object(), False)

    def test_cleanupConnectionImmediately(self):
        """
        L{_ExistingConnectionHelper.cleanupConnection} does nothing to the
        existing connection if called with C{immediate} set to C{True}.
        """
        helper = _ExistingConnectionHelper(object())
        helper.cleanupConnection(object(), True)