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
def test_permissionsShorthand(self) -> None:
    """
        L{Permissions}'s shorthand string is the RWX shorthand string for its
        user permission bits, group permission bits, and other permission bits
        concatenated together, without a space.
        """
    for u in range(0, 8):
        for g in range(0, 8):
            for o in range(0, 8):
                perm = filepath.Permissions(int('0o%d%d%d' % (u, g, o), 8))
                self.assertEqual(perm.shorthand(), ''.join((x.shorthand() for x in (perm.user, perm.group, perm.other))))
    self.assertEqual(filepath.Permissions(504).shorthand(), 'rwxrwx---')