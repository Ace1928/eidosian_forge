import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import time
from collections import Sequence
from contextlib import contextmanager
from errno import EINVAL, ENOENT
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
def _py2_fsencode(parts):
    return [part.encode(_py2_fs_encoding) if isinstance(part, unicode) else part for part in parts]