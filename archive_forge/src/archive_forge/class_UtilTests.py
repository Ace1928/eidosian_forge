import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
class UtilTests(TestCase):

    def testUniq(self):
        listWithDupes = ['a', 1, 'ab', 'a', 3, 4, 1, 2, 2, 4, 6]
        self.assertEqual(util.uniquify(listWithDupes), ['a', 1, 'ab', 3, 4, 2, 6])

    def testRaises(self):
        self.assertTrue(util.raises(ZeroDivisionError, divmod, 1, 0))
        self.assertFalse(util.raises(ZeroDivisionError, divmod, 0, 1))
        try:
            util.raises(TypeError, divmod, 1, 0)
        except ZeroDivisionError:
            pass
        else:
            raise FailTest("util.raises didn't raise when it should have")

    def test_uidFromNumericString(self):
        """
        When L{uidFromString} is called with a base-ten string representation
        of an integer, it returns the integer.
        """
        self.assertEqual(util.uidFromString('100'), 100)

    @skipIf(pwd is None, 'Username/UID conversion requires the pwd module.')
    def test_uidFromUsernameString(self):
        """
        When L{uidFromString} is called with a base-ten string representation
        of an integer, it returns the integer.
        """
        pwent = pwd.getpwuid(os.getuid())
        self.assertEqual(util.uidFromString(pwent.pw_name), pwent.pw_uid)

    def test_gidFromNumericString(self):
        """
        When L{gidFromString} is called with a base-ten string representation
        of an integer, it returns the integer.
        """
        self.assertEqual(util.gidFromString('100'), 100)

    @skipIf(grp is None, 'Group Name/GID conversion requires the grp module.')
    def test_gidFromGroupnameString(self):
        """
        When L{gidFromString} is called with a base-ten string representation
        of an integer, it returns the integer.
        """
        grent = grp.getgrgid(os.getgid())
        self.assertEqual(util.gidFromString(grent.gr_name), grent.gr_gid)