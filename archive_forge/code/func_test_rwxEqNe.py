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
def test_rwxEqNe(self) -> None:
    """
        L{RWX}'s created with the same booleans are equivalent.  If booleans
        are different, they are not equal.
        """
    for r in (True, False):
        for w in (True, False):
            for x in (True, False):
                self.assertEqual(filepath.RWX(r, w, x), filepath.RWX(r, w, x))
                self.assertNotUnequal(filepath.RWX(r, w, x), filepath.RWX(r, w, x))
    self.assertNotEqual(filepath.RWX(True, True, True), filepath.RWX(True, True, False))
    self.assertNotEqual(3, filepath.RWX(True, True, True))