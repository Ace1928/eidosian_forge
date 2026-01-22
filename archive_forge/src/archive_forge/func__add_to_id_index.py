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
def _add_to_id_index(self, id_index, entry_key):
    """Add this entry to the _id_index mapping."""
    file_id = entry_key[2]
    entry_key = static_tuple.StaticTuple.from_sequence(entry_key)
    if file_id not in id_index:
        id_index[file_id] = static_tuple.StaticTuple(entry_key)
    else:
        entry_keys = id_index[file_id]
        if entry_key not in entry_keys:
            id_index[file_id] = entry_keys + (entry_key,)