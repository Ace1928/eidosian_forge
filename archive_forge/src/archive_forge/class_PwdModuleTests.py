import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
class PwdModuleTests(TestCase, UserDatabaseTestsMixin):
    """
    L{PwdModuleTests} runs the tests defined by L{UserDatabaseTestsMixin}
    against the built-in C{pwd} module.  This serves to verify that
    L{UserDatabase} is really a fake of that API.
    """
    if pwd is None:
        skip = 'Cannot verify UserDatabase against pwd without pwd'
    else:
        database = pwd

    def setUp(self):
        self._users = iter(self.database.getpwall())
        self._uids = set()

    def getExistingUserInfo(self):
        """
        Read and return the next record from C{self._users}, filtering out
        any records with previously seen uid values (as these cannot be
        found with C{getpwuid} and only cause trouble).
        """
        while True:
            entry = next(self._users)
            uid = entry.pw_uid
            if uid not in self._uids:
                self._uids.add(uid)
                return entry