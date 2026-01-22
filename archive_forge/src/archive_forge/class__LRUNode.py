from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
class _LRUNode(Generic[K, V]):
    """This maintains the linked-list which is the lru internals."""
    __slots__ = ('prev', 'next_key', 'key', 'value', 'cleanup', 'size')
    prev: Optional['_LRUNode[K, V]']
    next_key: K
    size: Optional[int]

    def __init__(self, key: K, value: V, cleanup=None) -> None:
        self.prev = None
        self.next_key = _null_key
        self.key = key
        self.value = value
        self.cleanup = cleanup
        self.size = None

    def __repr__(self) -> str:
        if self.prev is None:
            prev_key = None
        else:
            prev_key = self.prev.key
        return '{}({!r} n:{!r} p:{!r})'.format(self.__class__.__name__, self.key, self.next_key, prev_key)

    def run_cleanup(self) -> None:
        if self.cleanup is not None:
            self.cleanup(self.key, self.value)
        self.cleanup = None
        del self.value