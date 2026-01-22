from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _getitems(self, file_ids):
    """Similar to get_entry, but lets you query for multiple.

        The returned order is undefined. And currently if an item doesn't
        exist, it isn't included in the output.
        """
    result = []
    remaining = []
    for file_id in file_ids:
        entry = self._fileid_to_entry_cache.get(file_id, None)
        if entry is None:
            remaining.append(file_id)
        else:
            result.append(entry)
    file_keys = [StaticTuple(f).intern() for f in remaining]
    for file_key, value in self.id_to_entry.iteritems(file_keys):
        entry = self._bytes_to_entry(value)
        result.append(entry)
        self._fileid_to_entry_cache[entry.file_id] = entry
    return result