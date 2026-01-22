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
def _get_id_index(self):
    """Get an id index of self._dirblocks.

        This maps from file_id => [(directory, name, file_id)] entries where
        that file_id appears in one of the trees.
        """
    if self._id_index is None:
        id_index = {}
        for key, tree_details in self._iter_entries():
            self._add_to_id_index(id_index, key)
        self._id_index = id_index
    return self._id_index