import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def _free_pending_blocks(self):
    while True:
        try:
            block = self._pending_free_blocks.pop()
        except IndexError:
            break
        self._add_free_block(block)
        self._remove_allocated_block(block)