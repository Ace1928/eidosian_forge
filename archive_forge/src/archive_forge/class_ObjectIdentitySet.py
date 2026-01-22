import collections
from typing import Any, Set
import weakref
class ObjectIdentitySet(collections.abc.MutableSet):
    """Like the built-in set, but compares objects with "is"."""
    __slots__ = ['_storage', '__weakref__']

    def __init__(self, *args):
        self._storage = set((self._wrap_key(obj) for obj in list(*args)))

    def __le__(self, other: Set[Any]) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) > len(other):
            return False
        for item in self._storage:
            if item not in other:
                return False
        return True

    def __ge__(self, other: Set[Any]) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        if len(self) < len(other):
            return False
        for item in other:
            if item not in self:
                return False
        return True

    @staticmethod
    def _from_storage(storage):
        result = ObjectIdentitySet()
        result._storage = storage
        return result

    def _wrap_key(self, key):
        return _ObjectIdentityWrapper(key)

    def __contains__(self, key):
        return self._wrap_key(key) in self._storage

    def discard(self, key):
        self._storage.discard(self._wrap_key(key))

    def add(self, key):
        self._storage.add(self._wrap_key(key))

    def update(self, items):
        self._storage.update([self._wrap_key(item) for item in items])

    def clear(self):
        self._storage.clear()

    def intersection(self, items):
        return self._storage.intersection([self._wrap_key(item) for item in items])

    def difference(self, items):
        return ObjectIdentitySet._from_storage(self._storage.difference([self._wrap_key(item) for item in items]))

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        keys = list(self._storage)
        for key in keys:
            yield key.unwrapped