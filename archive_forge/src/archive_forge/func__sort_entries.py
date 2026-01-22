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
def _sort_entries(self, entry_list):
    """Given a list of entries, sort them into the right order.

        This is done when constructing a new dirstate from trees - normally we
        try to keep everything in sorted blocks all the time, but sometimes
        it's easier to sort after the fact.
        """
    split_dirs = {}

    def _key(entry, _split_dirs=split_dirs, _st=static_tuple.StaticTuple):
        dirpath, fname, file_id = entry[0]
        try:
            split = _split_dirs[dirpath]
        except KeyError:
            split = _st.from_sequence(dirpath.split(b'/'))
            _split_dirs[dirpath] = split
        return _st(split, fname, file_id)
    return sorted(entry_list, key=_key)