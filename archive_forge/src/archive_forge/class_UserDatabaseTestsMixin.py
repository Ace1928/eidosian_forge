import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
class UserDatabaseTestsMixin:
    """
    L{UserDatabaseTestsMixin} defines tests which apply to any user database
    implementation.  Subclasses should mix it in, implement C{setUp} to create
    C{self.database} bound to a user database instance, and implement
    C{getExistingUserInfo} to return information about a user (such information
    should be unique per test method).
    """

    def test_getpwuid(self):
        """
        I{getpwuid} accepts a uid and returns the user record associated with
        it.
        """
        for i in range(2):
            username, password, uid, gid, gecos, dir, shell = self.getExistingUserInfo()
            entry = self.database.getpwuid(uid)
            self.assertEqual(entry.pw_name, username)
            self.assertEqual(entry.pw_passwd, password)
            self.assertEqual(entry.pw_uid, uid)
            self.assertEqual(entry.pw_gid, gid)
            self.assertEqual(entry.pw_gecos, gecos)
            self.assertEqual(entry.pw_dir, dir)
            self.assertEqual(entry.pw_shell, shell)

    def test_noSuchUID(self):
        """
        I{getpwuid} raises L{KeyError} when passed a uid which does not exist
        in the user database.
        """
        self.assertRaises(KeyError, self.database.getpwuid, INVALID_UID)

    def test_getpwnam(self):
        """
        I{getpwnam} accepts a username and returns the user record associated
        with it.
        """
        for i in range(2):
            username, password, uid, gid, gecos, dir, shell = self.getExistingUserInfo()
            entry = self.database.getpwnam(username)
            self.assertEqual(entry.pw_name, username)
            self.assertEqual(entry.pw_passwd, password)
            self.assertEqual(entry.pw_uid, uid)
            self.assertEqual(entry.pw_gid, gid)
            self.assertEqual(entry.pw_gecos, gecos)
            self.assertEqual(entry.pw_dir, dir)
            self.assertEqual(entry.pw_shell, shell)

    def test_getpwnamRejectsBytes(self):
        """
        L{getpwnam} rejects a non-L{str} username with an exception.
        """
        exc_type = TypeError
        if _PYPY:
            exc_type = Exception
        self.assertRaises(exc_type, self.database.getpwnam, b'i-am-bytes')

    def test_noSuchName(self):
        """
        I{getpwnam} raises L{KeyError} when passed a username which does not
        exist in the user database.
        """
        self.assertRaises(KeyError, self.database.getpwnam, 'nosuchuserexiststhenameistoolongandhas\x01inittoo')

    def test_recordLength(self):
        """
        The user record returned by I{getpwuid}, I{getpwnam}, and I{getpwall}
        has a length.
        """
        db = self.database
        username, password, uid, gid, gecos, dir, shell = self.getExistingUserInfo()
        for entry in [db.getpwuid(uid), db.getpwnam(username), db.getpwall()[0]]:
            self.assertIsInstance(len(entry), int)
            self.assertEqual(len(entry), 7)

    def test_recordIndexable(self):
        """
        The user record returned by I{getpwuid}, I{getpwnam}, and I{getpwall}
        is indexable, with successive indexes starting from 0 corresponding to
        the values of the C{pw_name}, C{pw_passwd}, C{pw_uid}, C{pw_gid},
        C{pw_gecos}, C{pw_dir}, and C{pw_shell} attributes, respectively.
        """
        db = self.database
        username, password, uid, gid, gecos, dir, shell = self.getExistingUserInfo()
        for entry in [db.getpwuid(uid), db.getpwnam(username), db.getpwall()[0]]:
            self.assertEqual(entry[0], username)
            self.assertEqual(entry[1], password)
            self.assertEqual(entry[2], uid)
            self.assertEqual(entry[3], gid)
            self.assertEqual(entry[4], gecos)
            self.assertEqual(entry[5], dir)
            self.assertEqual(entry[6], shell)
            self.assertEqual(len(entry), len(list(entry)))
            self.assertRaises(IndexError, getitem, entry, 7)