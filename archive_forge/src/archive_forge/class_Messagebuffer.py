import time
from collections import OrderedDict as _OrderedDict
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, MutableSet, Sequence
from heapq import heapify, heappop, heappush
from itertools import chain, count
from queue import Empty
from typing import Any, Dict, Iterable, List  # noqa
from .functional import first, uniq
from .text import match_case
class Messagebuffer(Evictable):
    """A buffer of pending messages."""
    Empty = Empty

    def __init__(self, maxsize, iterable=None, deque=deque):
        self.maxsize = maxsize
        self.data = deque(iterable or [])
        self._append = self.data.append
        self._pop = self.data.popleft
        self._len = self.data.__len__
        self._extend = self.data.extend

    def put(self, item):
        self._append(item)
        self.maxsize and self._evict()

    def extend(self, it):
        self._extend(it)
        self.maxsize and self._evict()

    def take(self, *default):
        try:
            return self._pop()
        except IndexError:
            if default:
                return default[0]
            raise self.Empty()

    def _pop_to_evict(self):
        return self.take()

    def __repr__(self):
        return f'<{type(self).__name__}: {len(self)}/{self.maxsize}>'

    def __iter__(self):
        while 1:
            try:
                yield self._pop()
            except IndexError:
                break

    def __len__(self):
        return self._len()

    def __contains__(self, item) -> bool:
        return item in self.data

    def __reversed__(self):
        return reversed(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def _evictcount(self):
        return len(self)