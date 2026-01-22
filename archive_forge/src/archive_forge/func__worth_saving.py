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
def _worth_saving(self):
    """Is it worth saving the dirstate or not?"""
    if self._header_state == DirState.IN_MEMORY_MODIFIED or self._dirblock_state == DirState.IN_MEMORY_MODIFIED:
        return True
    if self._dirblock_state == DirState.IN_MEMORY_HASH_MODIFIED:
        if self._worth_saving_limit == -1:
            return False
        if len(self._known_hash_changes) >= self._worth_saving_limit:
            return True
    return False