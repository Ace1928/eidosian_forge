from collections.abc import Mapping, Hashable
from itertools import chain
from typing import Generic, TypeVar
from pyrsistent._pvector import pvector
from pyrsistent._transformations import transform
def _reallocate(self):
    new_size = 2 * len(self._buckets_evolver)
    new_list = new_size * [None]
    buckets = self._buckets_evolver.persistent()
    for k, v in chain.from_iterable((x for x in buckets if x)):
        index = hash(k) % new_size
        if new_list[index]:
            new_list[index].append((k, v))
        else:
            new_list[index] = [(k, v)]
    self._buckets_evolver = pvector().evolver()
    self._buckets_evolver.extend(new_list)