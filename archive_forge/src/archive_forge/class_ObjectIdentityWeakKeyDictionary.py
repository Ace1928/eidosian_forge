import collections
from typing import Any, Set
import weakref
class ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
    """Like weakref.WeakKeyDictionary, but compares objects with "is"."""
    __slots__ = ['__weakref__']

    def _wrap_key(self, key):
        return _WeakObjectIdentityWrapper(key)

    def __len__(self):
        return len(list(self._storage))

    def __iter__(self):
        keys = self._storage.keys()
        for key in keys:
            unwrapped = key.unwrapped
            if unwrapped is None:
                del self[key]
            else:
                yield unwrapped