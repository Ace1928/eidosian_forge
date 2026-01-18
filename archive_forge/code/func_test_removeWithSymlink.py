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
@skipIf(symlinkSkip, 'Platform does not support symlinks')
def test_removeWithSymlink(self) -> None:
    """
        For a path which is a symbolic link, L{FilePath.remove} just deletes
        the link, not the target.
        """
    link = self.path.child(b'sub1.link')
    os.symlink(self.path.child(b'sub1').path, link.path)
    link.remove()
    self.assertFalse(link.exists())
    self.assertTrue(self.path.child(b'sub1').exists())