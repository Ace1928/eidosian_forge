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
def test_statinfoBitsAreNumbers(self) -> None:
    """
        Verify that file inode/device/nlinks/uid/gid stats are numbers in
        a POSIX environment
        """
    c = self.path.child(b'file1')
    for p in (self.path, c):
        self.assertIsInstance(p.getInodeNumber(), int)
        self.assertIsInstance(p.getDevice(), int)
        self.assertIsInstance(p.getNumberOfHardLinks(), int)
        self.assertIsInstance(p.getUserID(), int)
        self.assertIsInstance(p.getGroupID(), int)
    self.assertEqual(self.path.getUserID(), c.getUserID())
    self.assertEqual(self.path.getGroupID(), c.getGroupID())