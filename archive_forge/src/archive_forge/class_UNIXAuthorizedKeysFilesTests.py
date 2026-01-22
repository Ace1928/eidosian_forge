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
class UNIXAuthorizedKeysFilesTests(TestCase):
    """
    Tests for L{checkers.UNIXAuthorizedKeysFiles}.
    """
    skip = dependencySkip

    def setUp(self) -> None:
        self.path = FilePath(self.mktemp())
        assert isinstance(self.path.path, str)
        self.path.makedirs()
        self.userdb = UserDatabase()
        self.userdb.addUser('alice', 'password', 1, 2, 'alice lastname', self.path.path, '/bin/shell')
        self.sshDir = self.path.child('.ssh')
        self.sshDir.makedirs()
        authorizedKeys = self.sshDir.child('authorized_keys')
        authorizedKeys.setContent(b'key 1\nkey 2')
        self.expectedKeys = [b'key 1', b'key 2']

    def test_implementsInterface(self):
        """
        L{checkers.UNIXAuthorizedKeysFiles} implements
        L{checkers.IAuthorizedKeysDB}.
        """
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb)
        verifyObject(checkers.IAuthorizedKeysDB, keydb)

    def test_noKeysForUnauthorizedUser(self):
        """
        If the user is not in the user database provided to
        L{checkers.UNIXAuthorizedKeysFiles}, an empty iterator is returned
        by L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys}.
        """
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual([], list(keydb.getAuthorizedKeys(b'bob')))

    def test_allKeysInAllAuthorizedFilesForAuthorizedUser(self):
        """
        If the user is in the user database provided to
        L{checkers.UNIXAuthorizedKeysFiles}, an iterator with all the keys in
        C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2} is returned
        by L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys}.
        """
        self.sshDir.child('authorized_keys2').setContent(b'key 3')
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual(self.expectedKeys + [b'key 3'], list(keydb.getAuthorizedKeys(b'alice')))

    def test_ignoresNonexistantFile(self):
        """
        L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys} returns only
        the keys in C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2}
        if they exist.
        """
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual(self.expectedKeys, list(keydb.getAuthorizedKeys(b'alice')))

    def test_ignoresUnreadableFile(self):
        """
        L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys} returns only
        the keys in C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2}
        if they are readable.
        """
        self.sshDir.child('authorized_keys2').makedirs()
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual(self.expectedKeys, list(keydb.getAuthorizedKeys(b'alice')))