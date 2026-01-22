import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _read_all_roots(self):
    """Read the root pages.

        This is structured as a generator, so that the root records can be
        yielded up to whoever needs them without any buffering.
        """
    if not self._old_root_keys:
        self._new_queue = list(self._new_root_keys)
        return
    old_chks_to_enqueue = self._read_old_roots()
    new_keys = set(self._new_root_keys).difference(self._all_old_chks)
    new_prefixes = set()
    processed_new_refs = self._processed_new_refs
    processed_new_refs.update(new_keys)
    for record, node, prefix_refs, items in self._read_nodes_from_store(new_keys):
        prefix_refs = [p_r for p_r in prefix_refs if p_r[1] not in self._all_old_chks and p_r[1] not in processed_new_refs]
        refs = [p_r[1] for p_r in prefix_refs]
        new_prefixes.update([p_r[0] for p_r in prefix_refs])
        self._new_queue.extend(refs)
        new_items = [item for item in items if item not in self._all_old_items]
        self._new_item_queue.extend(new_items)
        new_prefixes.update([self._search_key_func(item[0]) for item in new_items])
        processed_new_refs.update(refs)
        yield record
    for prefix in list(new_prefixes):
        new_prefixes.update([prefix[:i] for i in range(1, len(prefix))])
    self._enqueue_old(new_prefixes, old_chks_to_enqueue)