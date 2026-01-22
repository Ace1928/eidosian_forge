from gitdb.db.base import (
from gitdb.util import LazyMixin
from gitdb.exc import (
from gitdb.pack import PackEntity
from functools import reduce
import os
import glob
def _pack_info(self, sha):
    """:return: tuple(entity, index) for an item at the given sha
        :param sha: 20 or 40 byte sha
        :raise BadObject:
        **Note:** This method is not thread-safe, but may be hit in multi-threaded
            operation. The worst thing that can happen though is a counter that
            was not incremented, or the list being in wrong order. So we safe
            the time for locking here, lets see how that goes"""
    if self._hit_count % self._sort_interval == 0:
        self._sort_entities()
    for item in self._entities:
        index = item[2](sha)
        if index is not None:
            item[0] += 1
            self._hit_count += 1
            return (item[1], index)
    raise BadObject(sha)