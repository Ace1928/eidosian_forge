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
def test_unicodeParent(self) -> None:
    """
        Calling C{parent} on a text-mode L{FilePath} will return a text-mode
        L{FilePath}.
        """
    fp = filepath.FilePath('./')
    parent = fp.parent()
    self.assertIsInstance(parent.path, str)