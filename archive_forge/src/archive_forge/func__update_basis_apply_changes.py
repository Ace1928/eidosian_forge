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
def _update_basis_apply_changes(self, changes):
    """Apply a sequence of changes to tree 1 during update_basis_by_delta.

        :param adds: A sequence of changes. Each change is a tuple:
            (path_utf8, path_utf8, file_id, (entry_details))
        """
    for old_path, new_path, file_id, new_details in changes:
        entry = self._get_entry(1, file_id, new_path)
        if entry[0] is None or entry[1][1][0] in (b'a', b'r'):
            self._raise_invalid(new_path, file_id, 'changed entry considered not present')
        entry[1][1] = new_details