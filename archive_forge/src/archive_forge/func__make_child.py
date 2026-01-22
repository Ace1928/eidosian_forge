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
def _make_child(self, args):
    drv, root, parts = self._parse_args(args)
    drv, root, parts = self._flavour.join_parsed_parts(self._drv, self._root, self._parts, drv, root, parts)
    return self._from_parsed_parts(drv, root, parts)