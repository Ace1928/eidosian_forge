import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _read_nodes_from_store(self, keys):
    as_st = StaticTuple.from_sequence
    stream = self._store.get_record_stream(keys, 'unordered', True)
    for record in stream:
        if self._pb is not None:
            self._pb.tick()
        if record.storage_kind == 'absent':
            raise errors.NoSuchRevision(self._store, record.key)
        bytes = record.get_bytes_as('fulltext')
        node = _deserialise(bytes, record.key, search_key_func=self._search_key_func)
        if isinstance(node, InternalNode):
            prefix_refs = list(node._items.items())
            items = []
        else:
            prefix_refs = []
            items = list(node._items.items())
        yield (record, node, prefix_refs, items)