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
def _discard_merge_parents(self):
    """Discard any parents trees beyond the first.

        Note that if this fails the dirstate is corrupted.

        After this function returns the dirstate contains 2 trees, neither of
        which are ghosted.
        """
    self._read_header_if_needed()
    parents = self.get_parent_ids()
    if len(parents) < 1:
        return
    self._read_dirblocks_if_needed()
    dead_patterns = {(b'a', b'r'), (b'a', b'a'), (b'r', b'r'), (b'r', b'a')}

    def iter_entries_removable():
        for block in self._dirblocks:
            deleted_positions = []
            for pos, entry in enumerate(block[1]):
                yield entry
                if (entry[1][0][0], entry[1][1][0]) in dead_patterns:
                    deleted_positions.append(pos)
            if deleted_positions:
                if len(deleted_positions) == len(block[1]):
                    del block[1][:]
                else:
                    for pos in reversed(deleted_positions):
                        del block[1][pos]
    if parents[0] in self.get_ghosts():
        empty_parent = [DirState.NULL_PARENT_DETAILS]
        for entry in iter_entries_removable():
            entry[1][1:] = empty_parent
    else:
        for entry in iter_entries_removable():
            del entry[1][2:]
    self._ghosts = []
    self._parents = [parents[0]]
    self._mark_modified(header_modified=True)