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
@skipIf(pwd is None, 'Username/UID conversion requires the pwd module.')
def test_uidFromUsernameString(self):
    """
        When L{uidFromString} is called with a base-ten string representation
        of an integer, it returns the integer.
        """
    pwent = pwd.getpwuid(os.getuid())
    self.assertEqual(util.uidFromString(pwent.pw_name), pwent.pw_uid)