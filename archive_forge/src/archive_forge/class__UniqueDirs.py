import os
import re
import fnmatch
import functools
from .util import convert_path
from .errors import DistutilsTemplateError, DistutilsInternalError
from ._log import log
class _UniqueDirs(set):
    """
    Exclude previously-seen dirs from walk results,
    avoiding infinite recursion.
    Ref https://bugs.python.org/issue44497.
    """

    def __call__(self, walk_item):
        """
        Given an item from an os.walk result, determine
        if the item represents a unique dir for this instance
        and if not, prevent further traversal.
        """
        base, dirs, files = walk_item
        stat = os.stat(base)
        candidate = (stat.st_dev, stat.st_ino)
        found = candidate in self
        if found:
            del dirs[:]
        self.add(candidate)
        return not found

    @classmethod
    def filter(cls, items):
        return filter(cls(), items)