import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import warnings
from _collections_abc import Sequence
from errno import ENOENT, ENOTDIR, EBADF, ELOOP
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
from urllib.parse import quote_from_bytes as urlquote_from_bytes
def with_name(self, name):
    """Return a new path with the file name changed."""
    if not self.name:
        raise ValueError('%r has an empty name' % (self,))
    drv, root, parts = self._flavour.parse_parts((name,))
    if not name or name[-1] in [self._flavour.sep, self._flavour.altsep] or drv or root or (len(parts) != 1):
        raise ValueError('Invalid name %r' % name)
    return self._from_parsed_parts(self._drv, self._root, self._parts[:-1] + [name])