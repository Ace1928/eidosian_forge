import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _add_free_block(self, block):
    arena, start, stop = block
    try:
        prev_block = self._stop_to_block[arena, start]
    except KeyError:
        pass
    else:
        start, _ = self._absorb(prev_block)
    try:
        next_block = self._start_to_block[arena, stop]
    except KeyError:
        pass
    else:
        _, stop = self._absorb(next_block)
    block = (arena, start, stop)
    length = stop - start
    try:
        self._len_to_seq[length].append(block)
    except KeyError:
        self._len_to_seq[length] = [block]
        bisect.insort(self._lengths, length)
    self._start_to_block[arena, start] = block
    self._stop_to_block[arena, stop] = block