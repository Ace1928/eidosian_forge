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
class AnonymousCheckerTests(TestCase):

    def test_isChecker(self):
        """
        Verifies that strcred.makeChecker('anonymous') returns an object
        that implements the L{ICredentialsChecker} interface.
        """
        checker = strcred.makeChecker('anonymous')
        self.assertTrue(checkers.ICredentialsChecker.providedBy(checker))
        self.assertIn(credentials.IAnonymous, checker.credentialInterfaces)

    def testAnonymousAccessSucceeds(self):
        """
        We can log in anonymously using this checker.
        """
        checker = strcred.makeChecker('anonymous')
        request = checker.requestAvatarId(credentials.Anonymous())

        def _gotAvatar(avatar):
            self.assertIdentical(checkers.ANONYMOUS, avatar)
        return request.addCallback(_gotAvatar)