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
def _bisect_recursive(self, paths):
    """Bisect for entries for all paths and their children.

        This will use bisect to find all records for the supplied paths. It
        will then continue to bisect for any records which are marked as
        directories. (and renames?)

        :param paths: A sorted list of (dir, name) pairs
             eg: [('', b'a'), ('', b'f'), ('a/b', b'c')]
        :return: A dictionary mapping (dir, name, file_id) => [tree_info]
        """
    found = {}
    found_dir_names = set()
    processed_dirs = set()
    newly_found = self._bisect(paths)
    while newly_found:
        pending_dirs = set()
        paths_to_search = set()
        for entry_list in newly_found.values():
            for dir_name_id, trees_info in entry_list:
                found[dir_name_id] = trees_info
                found_dir_names.add(dir_name_id[:2])
                is_dir = False
                for tree_info in trees_info:
                    minikind = tree_info[0]
                    if minikind == b'd':
                        if is_dir:
                            continue
                        subdir, name, file_id = dir_name_id
                        path = osutils.pathjoin(subdir, name)
                        is_dir = True
                        if path not in processed_dirs:
                            pending_dirs.add(path)
                    elif minikind == b'r':
                        dir_name = osutils.split(tree_info[1])
                        if dir_name[0] in pending_dirs:
                            continue
                        if dir_name not in found_dir_names:
                            paths_to_search.add(tree_info[1])
        newly_found = self._bisect(sorted(paths_to_search))
        newly_found.update(self._bisect_dirblocks(sorted(pending_dirs)))
        processed_dirs.update(pending_dirs)
    return found