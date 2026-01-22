from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
def _fileobj_lookup(self, fileobj):
    """Return a file descriptor from a file object.

        This wraps _fileobj_to_fd() to do an exhaustive search in case
        the object is invalid but we still have it in our map.  This
        is used by unregister() so we can unregister an object that
        was previously registered even if it is closed.  It is also
        used by _SelectorMapping.
        """
    try:
        return _fileobj_to_fd(fileobj)
    except ValueError:
        for key in self._fd_to_key.values():
            if key.fileobj is fileobj:
                return key.fd
        raise