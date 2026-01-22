import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class BaseFTPRealmTests(TestCase):
    """
    Tests for L{ftp.BaseFTPRealm}, a base class to help define L{IFTPShell}
    realms with different user home directory policies.
    """

    def test_interface(self):
        """
        L{ftp.BaseFTPRealm} implements L{IRealm}.
        """
        self.assertTrue(verifyClass(IRealm, ftp.BaseFTPRealm))

    def test_getHomeDirectory(self):
        """
        L{ftp.BaseFTPRealm} calls its C{getHomeDirectory} method with the
        avatarId being requested to determine the home directory for that
        avatar.
        """
        result = filepath.FilePath(self.mktemp())
        avatars = []

        class TestRealm(ftp.BaseFTPRealm):

            def getHomeDirectory(self, avatarId):
                avatars.append(avatarId)
                return result
        realm = TestRealm(self.mktemp())
        iface, avatar, logout = realm.requestAvatar('alice@example.com', None, ftp.IFTPShell)
        self.assertIsInstance(avatar, ftp.FTPShell)
        self.assertEqual(avatar.filesystemRoot, result)

    def test_anonymous(self):
        """
        L{ftp.BaseFTPRealm} returns an L{ftp.FTPAnonymousShell} instance for
        anonymous avatar requests.
        """
        anonymous = self.mktemp()
        realm = ftp.BaseFTPRealm(anonymous)
        iface, avatar, logout = realm.requestAvatar(checkers.ANONYMOUS, None, ftp.IFTPShell)
        self.assertIsInstance(avatar, ftp.FTPAnonymousShell)
        self.assertEqual(avatar.filesystemRoot, filepath.FilePath(anonymous))

    def test_notImplemented(self):
        """
        L{ftp.BaseFTPRealm.getHomeDirectory} should be overridden by a subclass
        and raises L{NotImplementedError} if it is not.
        """
        realm = ftp.BaseFTPRealm(self.mktemp())
        self.assertRaises(NotImplementedError, realm.getHomeDirectory, object())