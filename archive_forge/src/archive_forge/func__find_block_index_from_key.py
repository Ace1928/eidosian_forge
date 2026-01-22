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
def _find_block_index_from_key(self, key):
    """Find the dirblock index for a key.

        :return: The block index, True if the block for the key is present.
        """
    if key[0:2] == (b'', b''):
        return (0, True)
    try:
        if self._last_block_index is not None and self._dirblocks[self._last_block_index][0] == key[0]:
            return (self._last_block_index, True)
    except IndexError:
        pass
    block_index = bisect_dirblock(self._dirblocks, key[0], 1, cache=self._split_path_cache)
    present = block_index < len(self._dirblocks) and self._dirblocks[block_index][0] == key[0]
    self._last_block_index = block_index
    self._last_entry_index = -1
    return (block_index, present)