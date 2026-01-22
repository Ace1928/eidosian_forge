import sys
from collections.abc import MutableSet
from .compat import queue
class OrderedSet(MutableSet):
    """
        Implementation based on a doubly-linked link and an internal dictionary.
        This design gives :class:`OrderedSet` the same big-Oh running times as
        regular sets including O(1) adds, removes, and lookups as well as
        O(n) iteration.

        .. ADMONITION:: Implementation notes

                Runs on Python 2.6 or later (and runs on Python 3.0 or later
                without any modifications).

        :author: python@rcn.com (Raymond Hettinger)
        :url: http://code.activestate.com/recipes/576694/
        """

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[PREV]
            curr[NEXT] = end[PREV] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, _next = self.map.pop(key)
            prev[NEXT] = _next
            _next[PREV] = prev

    def __iter__(self):
        end = self.end
        curr = end[NEXT]
        while curr is not end:
            yield curr[KEY]
            curr = curr[NEXT]

    def __reversed__(self):
        end = self.end
        curr = end[PREV]
        while curr is not end:
            yield curr[KEY]
            curr = curr[PREV]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = next(reversed(self)) if last else next(iter(self))
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

    def __del__(self):
        self.clear()