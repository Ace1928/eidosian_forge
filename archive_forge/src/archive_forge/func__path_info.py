import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def _path_info(self, utf8_path, unicode_path):
    """Generate path_info for unicode_path.

        :return: None if unicode_path does not exist, or a path_info tuple.
        """
    abspath = self.tree.abspath(unicode_path)
    try:
        stat = os.lstat(abspath)
    except OSError as e:
        if e.errno == errno.ENOENT:
            return None
        else:
            raise
    utf8_basename = utf8_path.rsplit(b'/', 1)[-1]
    dir_info = (utf8_path, utf8_basename, osutils.file_kind_from_stat_mode(stat.st_mode), stat, abspath)
    if dir_info[2] == 'directory':
        if self.tree._directory_is_tree_reference(unicode_path):
            self.root_dir_info = self.root_dir_info[:2] + ('tree-reference',) + self.root_dir_info[3:]
    return dir_info