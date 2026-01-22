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
def _after_delta_check_parents(self, parents, index):
    """Check that parents required by the delta are all intact.

        :param parents: An iterable of (path_utf8, file_id) tuples which are
            required to be present in tree 'index' at path_utf8 with id file_id
            and be a directory.
        :param index: The column in the dirstate to check for parents in.
        """
    for dirname_utf8, file_id in parents:
        entry = self._get_entry(index, file_id, dirname_utf8)
        if entry[1] is None:
            self._raise_invalid(dirname_utf8.decode('utf8'), file_id, 'This parent is not present.')
        if entry[1][index][0] != b'd':
            self._raise_invalid(dirname_utf8.decode('utf8'), file_id, 'This parent is not a directory.')