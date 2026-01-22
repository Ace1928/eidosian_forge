from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
def _asFilesystemBytes(path: Union[bytes, str], encoding: Optional[str]='') -> bytes:
    """
    Return C{path} as a string of L{bytes} suitable for use on this system's
    filesystem.

    @param path: The path to be made suitable.
    @type path: L{bytes} or L{unicode}
    @param encoding: The encoding to use if coercing to L{bytes}. If none is
        given, L{sys.getfilesystemencoding} is used.

    @return: L{bytes}
    """
    if isinstance(path, bytes):
        return path
    else:
        if not encoding:
            encoding = sys.getfilesystemencoding()
        return path.encode(encoding, errors='surrogateescape')