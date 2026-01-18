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
def test_validSubdir(self) -> None:
    """
        Verify that a valid subdirectory will show up as a directory, but not as a
        file, not as a symlink, and be listable.
        """
    sub1 = self.path.child(b'sub1')
    self.assertTrue(sub1.exists(), 'This directory does exist.')
    self.assertTrue(sub1.isdir(), "It's a directory.")
    self.assertFalse(sub1.isfile(), "It's a directory.")
    self.assertFalse(sub1.islink(), "It's a directory.")
    self.assertEqual(sub1.listdir(), [b'file2'])