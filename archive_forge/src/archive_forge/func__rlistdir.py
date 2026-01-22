import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _rlistdir(dirname, dir_fd, dironly, include_hidden=False):
    names = _listdir(dirname, dir_fd, dironly)
    for x in names:
        if include_hidden or not _ishidden(x):
            yield x
            path = _join(dirname, x) if dirname else x
            for y in _rlistdir(path, dir_fd, dironly, include_hidden=include_hidden):
                yield _join(x, y)