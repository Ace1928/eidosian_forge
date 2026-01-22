from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def _check_scan_dir(self, fs, path, info, depth):
    """Check if a directory contents should be scanned."""
    if self.max_depth is not None and depth >= self.max_depth:
        return False
    return self.check_scan_dir(fs, path, info)