import abc
import weakref
from numba.core import errors
class RetargetCache:
    """Cache for retargeted dispatchers.

    The cache uses the original dispatcher as the key.
    """
    container_type = weakref.WeakKeyDictionary

    def __init__(self):
        self._cache = self.container_type()
        self._stat_hit = 0
        self._stat_miss = 0

    def save_cache(self, orig_disp, new_disp):
        """Save a dispatcher associated with the given key.
        """
        self._cache[orig_disp] = new_disp

    def load_cache(self, orig_disp):
        """Load a dispatcher associated with the given key.
        """
        out = self._cache.get(orig_disp)
        if out is None:
            self._stat_miss += 1
        else:
            self._stat_hit += 1
        return out

    def items(self):
        """Returns the contents of the cache.
        """
        return self._cache.items()

    def stats(self):
        """Returns stats regarding cache hit/miss.
        """
        return {'hit': self._stat_hit, 'miss': self._stat_miss}