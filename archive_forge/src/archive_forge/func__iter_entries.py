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
def _iter_entries(self):
    """Iterate over all the entries in the dirstate.

        Each yelt item is an entry in the standard format described in the
        docstring of breezy.dirstate.
        """
    self._read_dirblocks_if_needed()
    for directory in self._dirblocks:
        yield from directory[1]