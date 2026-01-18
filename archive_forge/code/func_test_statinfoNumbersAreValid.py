from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
@skipIf(platform.isWindows(), 'Test does not run on Windows')
def test_statinfoNumbersAreValid(self) -> None:
    """
        Verify that the right numbers come back from the right accessor methods
        for file inode/device/nlinks/uid/gid (in a POSIX environment)
        """

    class FakeStat:
        st_ino = 200
        st_dev = 300
        st_nlink = 400
        st_uid = 500
        st_gid = 600
    fake = FakeStat()

    def fakeRestat(*args: object, **kwargs: object) -> None:
        self.path._statinfo = fake
    self.path.restat = fakeRestat
    self.path._statinfo = None
    self.assertEqual(self.path.getInodeNumber(), fake.st_ino)
    self.assertEqual(self.path.getDevice(), fake.st_dev)
    self.assertEqual(self.path.getNumberOfHardLinks(), fake.st_nlink)
    self.assertEqual(self.path.getUserID(), fake.st_uid)
    self.assertEqual(self.path.getGroupID(), fake.st_gid)