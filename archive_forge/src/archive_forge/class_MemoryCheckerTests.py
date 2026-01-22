import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
class MemoryCheckerTests(TestCase):

    def setUp(self):
        self.admin = credentials.UsernamePassword('admin', 'asdf')
        self.alice = credentials.UsernamePassword('alice', 'foo')
        self.badPass = credentials.UsernamePassword('alice', 'foobar')
        self.badUser = credentials.UsernamePassword('x', 'yz')
        self.checker = strcred.makeChecker('memory:admin:asdf:alice:foo')

    def test_isChecker(self):
        """
        Verifies that strcred.makeChecker('memory') returns an object
        that implements the L{ICredentialsChecker} interface.
        """
        self.assertTrue(checkers.ICredentialsChecker.providedBy(self.checker))
        self.assertIn(credentials.IUsernamePassword, self.checker.credentialInterfaces)

    def test_badFormatArgString(self):
        """
        An argument string which does not contain user:pass pairs
        (i.e., an odd number of ':' characters) raises an exception.
        """
        self.assertRaises(strcred.InvalidAuthArgumentString, strcred.makeChecker, 'memory:a:b:c')

    def test_memoryCheckerSucceeds(self):
        """
        The checker works with valid credentials.
        """

        def _gotAvatar(username):
            self.assertEqual(username, self.admin.username)
        return self.checker.requestAvatarId(self.admin).addCallback(_gotAvatar)

    def test_memoryCheckerFailsUsername(self):
        """
        The checker fails with an invalid username.
        """
        return self.assertFailure(self.checker.requestAvatarId(self.badUser), error.UnauthorizedLogin)

    def test_memoryCheckerFailsPassword(self):
        """
        The checker fails with an invalid password.
        """
        return self.assertFailure(self.checker.requestAvatarId(self.badPass), error.UnauthorizedLogin)