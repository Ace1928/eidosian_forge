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
class InMemorySSHKeyDBTests(TestCase):
    """
    Tests for L{checkers.InMemorySSHKeyDB}
    """
    skip = dependencySkip

    def test_implementsInterface(self):
        """
        L{checkers.InMemorySSHKeyDB} implements
        L{checkers.IAuthorizedKeysDB}
        """
        keydb = checkers.InMemorySSHKeyDB({b'alice': [b'key']})
        verifyObject(checkers.IAuthorizedKeysDB, keydb)

    def test_noKeysForUnauthorizedUser(self):
        """
        If the user is not in the mapping provided to
        L{checkers.InMemorySSHKeyDB}, an empty iterator is returned
        by L{checkers.InMemorySSHKeyDB.getAuthorizedKeys}
        """
        keydb = checkers.InMemorySSHKeyDB({b'alice': [b'keys']})
        self.assertEqual([], list(keydb.getAuthorizedKeys(b'bob')))

    def test_allKeysForAuthorizedUser(self):
        """
        If the user is in the mapping provided to
        L{checkers.InMemorySSHKeyDB}, an iterator with all the keys
        is returned by L{checkers.InMemorySSHKeyDB.getAuthorizedKeys}
        """
        keydb = checkers.InMemorySSHKeyDB({b'alice': [b'a', b'b']})
        self.assertEqual([b'a', b'b'], list(keydb.getAuthorizedKeys(b'alice')))