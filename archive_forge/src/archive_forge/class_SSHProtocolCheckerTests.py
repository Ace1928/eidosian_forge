import os
from base64 import encodebytes
from collections import namedtuple
from io import BytesIO
from typing import Optional
from zope.interface.verify import verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet.defer import Deferred
from twisted.python import util
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
class SSHProtocolCheckerTests(TestCase):
    """
    Tests for L{SSHProtocolChecker}.
    """
    skip = dependencySkip

    def test_registerChecker(self):
        """
        L{SSHProcotolChecker.registerChecker} should add the given checker to
        the list of registered checkers.
        """
        checker = checkers.SSHProtocolChecker()
        self.assertEqual(checker.credentialInterfaces, [])
        checker.registerChecker(checkers.SSHPublicKeyDatabase())
        self.assertEqual(checker.credentialInterfaces, [ISSHPrivateKey])
        self.assertIsInstance(checker.checkers[ISSHPrivateKey], checkers.SSHPublicKeyDatabase)

    def test_registerCheckerWithInterface(self):
        """
        If a specific interface is passed into
        L{SSHProtocolChecker.registerChecker}, that interface should be
        registered instead of what the checker specifies in
        credentialIntefaces.
        """
        checker = checkers.SSHProtocolChecker()
        self.assertEqual(checker.credentialInterfaces, [])
        checker.registerChecker(checkers.SSHPublicKeyDatabase(), IUsernamePassword)
        self.assertEqual(checker.credentialInterfaces, [IUsernamePassword])
        self.assertIsInstance(checker.checkers[IUsernamePassword], checkers.SSHPublicKeyDatabase)

    def test_requestAvatarId(self):
        """
        L{SSHProtocolChecker.requestAvatarId} should defer to one if its
        registered checkers to authenticate a user.
        """
        checker = checkers.SSHProtocolChecker()
        passwordDatabase = InMemoryUsernamePasswordDatabaseDontUse()
        passwordDatabase.addUser(b'test', b'test')
        checker.registerChecker(passwordDatabase)
        d = checker.requestAvatarId(UsernamePassword(b'test', b'test'))

        def _callback(avatarId):
            self.assertEqual(avatarId, b'test')
        return d.addCallback(_callback)

    def test_requestAvatarIdWithNotEnoughAuthentication(self):
        """
        If the client indicates that it is never satisfied, by always returning
        False from _areDone, then L{SSHProtocolChecker} should raise
        L{NotEnoughAuthentication}.
        """
        checker = checkers.SSHProtocolChecker()

        def _areDone(avatarId):
            return False
        self.patch(checker, 'areDone', _areDone)
        passwordDatabase = InMemoryUsernamePasswordDatabaseDontUse()
        passwordDatabase.addUser(b'test', b'test')
        checker.registerChecker(passwordDatabase)
        d = checker.requestAvatarId(UsernamePassword(b'test', b'test'))
        return self.assertFailure(d, NotEnoughAuthentication)

    def test_requestAvatarIdInvalidCredential(self):
        """
        If the passed credentials aren't handled by any registered checker,
        L{SSHProtocolChecker} should raise L{UnhandledCredentials}.
        """
        checker = checkers.SSHProtocolChecker()
        d = checker.requestAvatarId(UsernamePassword(b'test', b'test'))
        return self.assertFailure(d, UnhandledCredentials)

    def test_areDone(self):
        """
        The default L{SSHProcotolChecker.areDone} should simply return True.
        """
        self.assertTrue(checkers.SSHProtocolChecker().areDone(None))