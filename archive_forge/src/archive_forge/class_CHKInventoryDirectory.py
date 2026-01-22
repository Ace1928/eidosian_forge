from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class CHKInventoryDirectory(InventoryDirectory):
    """A directory in an inventory."""
    __slots__ = ['_children', '_chk_inventory']

    def __init__(self, file_id, name, parent_id, chk_inventory):
        InventoryEntry.__init__(self, file_id, name, parent_id)
        self._children = None
        self._chk_inventory = chk_inventory

    @property
    def children(self):
        """Access the list of children of this directory.

        With a parent_id_basename_to_file_id index, loads all the children,
        without loads the entire index. Without is bad. A more sophisticated
        proxy object might be nice, to allow partial loading of children as
        well when specific names are accessed. (So path traversal can be
        written in the obvious way but not examine siblings.).
        """
        if self._children is not None:
            return self._children
        if self._chk_inventory.parent_id_basename_to_file_id is None:
            raise AssertionError('Inventories without parent_id_basename_to_file_id are no longer supported')
        result = {}
        parent_id_index = self._chk_inventory.parent_id_basename_to_file_id
        child_keys = set()
        for (parent_id, name_utf8), file_id in parent_id_index.iteritems(key_filter=[StaticTuple(self.file_id)]):
            child_keys.add(StaticTuple(file_id))
        cached = set()
        for file_id_key in child_keys:
            entry = self._chk_inventory._fileid_to_entry_cache.get(file_id_key[0], None)
            if entry is not None:
                result[entry.name] = entry
                cached.add(file_id_key)
        child_keys.difference_update(cached)
        id_to_entry = self._chk_inventory.id_to_entry
        for file_id_key, bytes in id_to_entry.iteritems(child_keys):
            entry = self._chk_inventory._bytes_to_entry(bytes)
            result[entry.name] = entry
            self._chk_inventory._fileid_to_entry_cache[file_id_key[0]] = entry
        self._children = result
        return result