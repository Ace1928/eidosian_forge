import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _lexists(pathname, dir_fd):
    if dir_fd is None:
        return os.path.lexists(pathname)
    try:
        os.lstat(pathname, dir_fd=dir_fd)
    except (OSError, ValueError):
        return False
    else:
        return True