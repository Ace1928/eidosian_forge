import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class LateReadError:
    """A helper for transports which pretends to be a readable file.

    When read() is called, errors.ReadError is raised.
    """

    def __init__(self, path):
        self._path = path

    def close(self):
        """a no-op - do nothing."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except:
            if exc_type is None:
                raise
        return False

    def _fail(self):
        """Raise ReadError."""
        raise errors.ReadError(self._path)

    def __iter__(self):
        self._fail()

    def read(self, count=-1):
        self._fail()

    def readlines(self):
        self._fail()