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
def _coerceToFilesystemEncoding(path: AnyStr, newpath: Union[bytes, str], encoding: Optional[str]=None) -> AnyStr:
    """
    Return a C{newpath} that is suitable for joining to C{path}.

    @param path: The path that it should be suitable for joining to.
    @param newpath: The new portion of the path to be coerced if needed.
    @param encoding: If coerced, the encoding that will be used.
    """
    if isinstance(path, bytes):
        return _asFilesystemBytes(newpath, encoding=encoding)
    else:
        return _asFilesystemText(newpath, encoding=encoding)