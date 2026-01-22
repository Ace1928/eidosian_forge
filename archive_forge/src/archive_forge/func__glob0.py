import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _glob0(dirname, basename, dir_fd, dironly, include_hidden=False):
    if basename:
        if _lexists(_join(dirname, basename), dir_fd):
            return [basename]
    elif _isdir(dirname, dir_fd):
        return [basename]
    return []