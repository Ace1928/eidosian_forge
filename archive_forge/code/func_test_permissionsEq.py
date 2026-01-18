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
def test_permissionsEq(self) -> None:
    """
        Two L{Permissions}'s that are created with the same bitmask
        are equivalent
        """
    self.assertEqual(filepath.Permissions(511), filepath.Permissions(511))
    self.assertNotUnequal(filepath.Permissions(511), filepath.Permissions(511))
    self.assertNotEqual(filepath.Permissions(511), filepath.Permissions(448))
    self.assertNotEqual(3, filepath.Permissions(511))