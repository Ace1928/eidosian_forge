import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
class LRUMemo:
    """
    A memoizing mapping that retains `capacity` deleted items

    The memo tracks retained items by their access order; once `capacity` items
    are retained, the least recently used item is discarded.
    """

    def __init__(self, capacity):
        self._capacity = capacity
        self._active = {}
        self._memory = collections.OrderedDict()

    def __getitem__(self, key):
        try:
            return self._active[key]
        except KeyError:
            self._memory.move_to_end(key)
            return self._memory[key]

    def __setitem__(self, key, value):
        self._memory.pop(key, None)
        self._active[key] = value

    def __delitem__(self, key):
        try:
            value = self._active.pop(key)
        except KeyError:
            pass
        else:
            while len(self._memory) >= self._capacity:
                self._memory.popitem(last=False)
            self._memory[key] = value

    def clear(self):
        self._active.clear()
        self._memory.clear()