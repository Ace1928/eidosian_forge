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

        Move self to destination - basically renaming self to whatever
        destination is named.

        If destination is an already-existing directory,
        moves all children to destination if destination is empty.  If
        destination is a non-empty directory, or destination is a file, an
        OSError will be raised.

        If moving between filesystems, self needs to be copied, and everything
        that applies to copyTo applies to moveTo.

        @param destination: the destination (a FilePath) to which self
            should be copied
        @param followLinks: whether symlinks in self should be treated as links
            or as their targets (only applicable when moving between
            filesystems)
        