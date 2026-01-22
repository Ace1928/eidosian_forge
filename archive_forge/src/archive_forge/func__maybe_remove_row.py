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
def _maybe_remove_row(self, block, index, id_index):
    """Remove index if it is absent or relocated across the row.

        id_index is updated accordingly.
        :return: True if we removed the row, False otherwise
        """
    present_in_row = False
    entry = block[index]
    for column in entry[1]:
        if column[0] not in (b'a', b'r'):
            present_in_row = True
            break
    if not present_in_row:
        block.pop(index)
        self._remove_from_id_index(id_index, entry[0])
        return True
    return False