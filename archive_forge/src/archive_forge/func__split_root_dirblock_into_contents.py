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
def _split_root_dirblock_into_contents(self):
    """Split the root dirblocks into root and contents-of-root.

        After parsing by path, we end up with root entries and contents-of-root
        entries in the same block. This loop splits them out again.
        """
    if self._dirblocks[1] != (b'', []):
        raise ValueError('bad dirblock start {!r}'.format(self._dirblocks[1]))
    root_block = []
    contents_of_root_block = []
    for entry in self._dirblocks[0][1]:
        if not entry[0][1]:
            root_block.append(entry)
        else:
            contents_of_root_block.append(entry)
    self._dirblocks[0] = (b'', root_block)
    self._dirblocks[1] = (b'', contents_of_root_block)