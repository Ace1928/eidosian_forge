from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def _walk_lru(self) -> Iterator[_LRUNode[K, V]]:
    """Walk the LRU list, only meant to be used in tests."""
    node = self._most_recently_used
    if node is not None:
        if node.prev is not None:
            raise AssertionError(f'the _most_recently_used entry is not supposed to have a previous entry {node}')
    while node is not None:
        if node.next_key is _null_key:
            if node is not self._least_recently_used:
                raise AssertionError(f'only the last node should have no next value: {node}')
            node_next = None
        else:
            node_next = self._cache[node.next_key]
            if node_next.prev is not node:
                raise AssertionError(f'inconsistency found, node.next.prev != node: {node}')
        if node.prev is None:
            if node is not self._most_recently_used:
                raise AssertionError(f'only the _most_recently_used should not have a previous node: {node}')
        elif node.prev.next_key != node.key:
            raise AssertionError(f'inconsistency found, node.prev.next != node: {node}')
        yield node
        node = node_next