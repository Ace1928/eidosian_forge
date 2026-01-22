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
def _entries_to_current_state(self, new_entries):
    """Load new_entries into self.dirblocks.

        Process new_entries into the current state object, making them the active
        state.  The entries are grouped together by directory to form dirblocks.

        :param new_entries: A sorted list of entries. This function does not sort
            to prevent unneeded overhead when callers have a sorted list already.
        :return: Nothing.
        """
    if new_entries[0][0][0:2] != (b'', b''):
        raise AssertionError('Missing root row {!r}'.format(new_entries[0][0]))
    self._dirblocks = [(b'', []), (b'', [])]
    current_block = self._dirblocks[0][1]
    current_dirname = b''
    root_key = (b'', b'')
    append_entry = current_block.append
    for entry in new_entries:
        if entry[0][0] != current_dirname:
            current_block = []
            current_dirname = entry[0][0]
            self._dirblocks.append((current_dirname, current_block))
            append_entry = current_block.append
        append_entry(entry)
    self._split_root_dirblock_into_contents()