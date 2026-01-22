import functools
import math
import os
import random
import threading
import time
from .core import ENOVAL, args_to_key, full_name
class Averager:
    """Recipe for calculating a running average.

    Sometimes known as "online statistics," the running average maintains the
    total and count. The average can then be calculated at any time.

    Assumes the key will not be evicted. Set the eviction policy to 'none' on
    the cache to guarantee the key is not evicted.

    >>> import diskcache
    >>> cache = diskcache.FanoutCache()
    >>> ave = Averager(cache, 'latency')
    >>> ave.add(0.080)
    >>> ave.add(0.120)
    >>> ave.get()
    0.1
    >>> ave.add(0.160)
    >>> ave.pop()
    0.12
    >>> print(ave.get())
    None

    """

    def __init__(self, cache, key, expire=None, tag=None):
        self._cache = cache
        self._key = key
        self._expire = expire
        self._tag = tag

    def add(self, value):
        """Add `value` to average."""
        with self._cache.transact(retry=True):
            total, count = self._cache.get(self._key, default=(0.0, 0))
            total += value
            count += 1
            self._cache.set(self._key, (total, count), expire=self._expire, tag=self._tag)

    def get(self):
        """Get current average or return `None` if count equals zero."""
        total, count = self._cache.get(self._key, default=(0.0, 0), retry=True)
        return None if count == 0 else total / count

    def pop(self):
        """Return current average and delete key."""
        total, count = self._cache.pop(self._key, default=(0.0, 0), retry=True)
        return None if count == 0 else total / count