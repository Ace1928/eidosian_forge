import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _iter_nodes(self, store, key_filter=None, batch_size=None):
    """Iterate over node objects which match key_filter.

        :param store: A store to use for accessing content.
        :param key_filter: A key filter to filter nodes. Only nodes that might
            contain a key in key_filter will be returned.
        :param batch_size: If not None, then we will return the nodes that had
            to be read using get_record_stream in batches, rather than reading
            them all at once.
        :return: An iterable of nodes. This function does not have to be fully
            consumed.  (There will be no pending I/O when items are being returned.)
        """
    keys = {}
    shortcut = False
    if key_filter is None:
        shortcut = True
        for prefix, node in self._items.items():
            if node.__class__ is StaticTuple:
                keys[node] = (prefix, None)
            else:
                yield (node, None)
    elif len(key_filter) == 1:
        for key in key_filter:
            break
        search_prefix = self._search_prefix_filter(key)
        if len(search_prefix) == self._node_width:
            shortcut = True
            try:
                node = self._items[search_prefix]
            except KeyError:
                return
            if node.__class__ is StaticTuple:
                keys[node] = (search_prefix, [key])
            else:
                yield (node, [key])
                return
    if not shortcut:
        prefix_to_keys = {}
        length_filters = {}
        for key in key_filter:
            search_prefix = self._search_prefix_filter(key)
            length_filter = length_filters.setdefault(len(search_prefix), set())
            length_filter.add(search_prefix)
            prefix_to_keys.setdefault(search_prefix, []).append(key)
        if self._node_width in length_filters and len(length_filters) == 1:
            search_prefixes = length_filters[self._node_width]
            for search_prefix in search_prefixes:
                try:
                    node = self._items[search_prefix]
                except KeyError:
                    continue
                node_key_filter = prefix_to_keys[search_prefix]
                if node.__class__ is StaticTuple:
                    keys[node] = (search_prefix, node_key_filter)
                else:
                    yield (node, node_key_filter)
        else:
            length_filters_itemview = length_filters.items()
            for prefix, node in self._items.items():
                node_key_filter = []
                for length, length_filter in length_filters_itemview:
                    sub_prefix = prefix[:length]
                    if sub_prefix in length_filter:
                        node_key_filter.extend(prefix_to_keys[sub_prefix])
                if node_key_filter:
                    if node.__class__ is StaticTuple:
                        keys[node] = (prefix, node_key_filter)
                    else:
                        yield (node, node_key_filter)
    if keys:
        found_keys = set()
        for key in keys:
            try:
                bytes = _get_cache()[key]
            except KeyError:
                continue
            else:
                node = _deserialise(bytes, key, search_key_func=self._search_key_func)
                prefix, node_key_filter = keys[key]
                self._items[prefix] = node
                found_keys.add(key)
                yield (node, node_key_filter)
        for key in found_keys:
            del keys[key]
    if keys:
        if batch_size is None:
            batch_size = len(keys)
        key_order = list(keys)
        for batch_start in range(0, len(key_order), batch_size):
            batch = key_order[batch_start:batch_start + batch_size]
            stream = store.get_record_stream(batch, 'unordered', True)
            node_and_filters = []
            for record in stream:
                bytes = record.get_bytes_as('fulltext')
                node = _deserialise(bytes, record.key, search_key_func=self._search_key_func)
                prefix, node_key_filter = keys[record.key]
                node_and_filters.append((node, node_key_filter))
                self._items[prefix] = node
                _get_cache()[record.key] = bytes
            yield from node_and_filters