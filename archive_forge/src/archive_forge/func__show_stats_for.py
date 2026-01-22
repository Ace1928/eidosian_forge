import atexit
import os
import shutil
import tempfile
import weakref
from fastimport.reftracker import RefTracker
from ... import lru_cache, trace
from . import branch_mapper
from .helpers import single_plural
def _show_stats_for(self, a_dict, label, note, tuple_key=False):
    """Dump statistics about a given dictionary.

        By the key and value need to support len().
        """
    count = len(a_dict)
    if tuple_key:
        size = sum(map(len, (''.join(k) for k in a_dict)))
    else:
        size = sum(map(len, a_dict))
    size += sum(map(len, a_dict.values()))
    size = size * 1.0 / 1024
    unit = 'K'
    if size > 1024:
        size = size / 1024
        unit = 'M'
        if size > 1024:
            size = size / 1024
            unit = 'G'
    note('    %-12s: %8.1f %s (%d %s)' % (label, size, unit, count, single_plural(count, 'item', 'items')))