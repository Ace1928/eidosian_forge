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
@classmethod
def _from_parts(cls, args):
    self = object.__new__(cls)
    drv, root, parts = self._parse_args(args)
    self._drv = drv
    self._root = root
    self._parts = parts
    return self