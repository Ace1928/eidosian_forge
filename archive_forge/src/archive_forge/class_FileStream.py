import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class FileStream:
    """Base class for FileStreams."""

    def __init__(self, transport, relpath):
        """Create a FileStream for relpath on transport."""
        self.transport = transport
        self.relpath = relpath

    def _close(self):
        """A hook point for subclasses that need to take action on close."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
        return False

    def close(self, want_fdatasync=False):
        if want_fdatasync:
            try:
                self.fdatasync()
            except errors.TransportNotPossible:
                pass
        self._close()
        del _file_streams[self.transport.abspath(self.relpath)]

    def fdatasync(self):
        """Force data out to physical disk if possible.

        :raises TransportNotPossible: If this transport has no way to
            flush to disk.
        """
        raise errors.TransportNotPossible('{} cannot fdatasync'.format(self.transport))