from __future__ import annotations
import errno
import os
import time
from typing import (
from zipfile import ZipFile
from zope.interface import implementer
from typing_extensions import Literal, Self
from twisted.python.compat import cmp, comparable
from twisted.python.filepath import (
def _nativeParent(self) -> Union[ZipPath[_ZipStr, _ArchiveStr], ZipArchive[_ArchiveStr]]:
    """
        Return parent, discarding our own encoding in favor of whatever the
        archive's is.
        """
    splitup = self.pathInArchive.split(self.sep)
    if len(splitup) == 1:
        return self.archive
    return ZipPath(self.archive, self.sep.join(splitup[:-1]))