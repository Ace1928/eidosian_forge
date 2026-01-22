import os
import sys
import stat
import fnmatch
import collections
import errno
def _stat(fn):
    return fn.stat() if isinstance(fn, os.DirEntry) else os.stat(fn)