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
def test_realpathCyclicalSymlink(self) -> None:
    """
        L{FilePath.realpath} raises L{filepath.LinkError} if the path is a
        symbolic link which is part of a cycle.
        """
    os.symlink(self.path.child(b'link1').path, self.path.child(b'link2').path)
    os.symlink(self.path.child(b'link2').path, self.path.child(b'link1').path)
    self.assertRaises(filepath.LinkError, self.path.child(b'link2').realpath)