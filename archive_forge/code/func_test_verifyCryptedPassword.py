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
def test_verifyCryptedPassword(self):
    """
        L{verifyCryptedPassword} returns C{True} if the plaintext password
        passed to it matches the encrypted password passed to it.
        """
    password = 'secret string'
    salt = 'salty'
    crypted = crypt.crypt(password, salt)
    self.assertTrue(checkers.verifyCryptedPassword(crypted, password), '{!r} supposed to be valid encrypted password for {!r}'.format(crypted, password))