import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
def _check_parent(self, _abspath):
    dir = os.path.dirname(_abspath)
    if dir != '/':
        if dir not in self._dirs:
            raise NoSuchFile(_abspath)