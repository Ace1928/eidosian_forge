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