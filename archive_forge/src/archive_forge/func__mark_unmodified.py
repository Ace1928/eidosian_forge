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
def _mark_unmodified(self):
    """Mark this dirstate as unmodified."""
    self._header_state = DirState.IN_MEMORY_UNMODIFIED
    self._dirblock_state = DirState.IN_MEMORY_UNMODIFIED
    self._known_hash_changes = set()