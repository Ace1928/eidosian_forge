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
def test_verifyCryptedPasswordMD5(self):
    """
        L{verifyCryptedPassword} returns True if the provided cleartext password
        matches the provided MD5 password hash.
        """
    password = 'password'
    salt = '$1$salt'
    crypted = crypt.crypt(password, salt)
    self.assertTrue(checkers.verifyCryptedPassword(crypted, password), '{!r} supposed to be valid encrypted password for {}'.format(crypted, password))