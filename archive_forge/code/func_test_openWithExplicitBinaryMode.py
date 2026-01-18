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
def test_openWithExplicitBinaryMode(self) -> None:
    """
        Due to a bug in Python 2.7 on Windows including multiple 'b'
        characters in the mode passed to the built-in open() will cause an
        error.  FilePath.open() ensures that only a single 'b' character is
        included in the mode passed to the built-in open().

        See http://bugs.python.org/issue7686 for details about the bug.
        """
    writer = self.path.child(b'explicit-binary')
    opener = writer.open('wb')
    with opener as file:
        file.write(b'abc\ndef')
    self.assertTrue(writer.exists)