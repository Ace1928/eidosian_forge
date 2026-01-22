import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
def _resolve_symlinks(self, relpath):
    path = self._abspath(relpath)
    while path in self._symlinks.keys():
        path = self._symlinks[path]
    return path