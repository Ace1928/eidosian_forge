import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _new_arena(self, size):
    length = self._roundup(max(self._size, size), mmap.PAGESIZE)
    if self._size < self._DOUBLE_ARENA_SIZE_UNTIL:
        self._size *= 2
    util.info('allocating a new mmap of length %d', length)
    arena = Arena(length)
    self._arenas.append(arena)
    return (arena, 0, length)