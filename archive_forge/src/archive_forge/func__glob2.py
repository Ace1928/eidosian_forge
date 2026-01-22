import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def _glob2(dirname, pattern, dir_fd, dironly, include_hidden=False):
    assert _isrecursive(pattern)
    yield pattern[:0]
    yield from _rlistdir(dirname, dir_fd, dironly, include_hidden=include_hidden)