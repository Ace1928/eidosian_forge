from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
class ExpiringLRUCache(Cache):
    """ Implements a pseudo-LRU algorithm (CLOCK) with expiration times

    The Clock algorithm is not kept strictly to improve performance, e.g. to
    allow get() and invalidate() to work without acquiring the lock.
    """

    def __init__(self, size, default_timeout=_DEFAULT_TIMEOUT):
        self.default_timeout = default_timeout
        size = int(size)
        if size < 1:
            raise ValueError('size must be >0')
        self.size = size
        self.lock = threading.Lock()
        self.hand = 0
        self.maxpos = size - 1
        self.clock_keys = None
        self.clock_refs = None
        self.data = None
        self.evictions = 0
        self.hits = 0
        self.misses = 0
        self.lookups = 0
        self.clear()

    def clear(self):
        """Remove all entries from the cache"""
        with self.lock:
            self.data = {}
            size = self.size
            self.clock_keys = [_MARKER] * size
            self.clock_refs = [False] * size
            self.hand = 0
            self.evictions = 0
            self.hits = 0
            self.misses = 0
            self.lookups = 0

    def get(self, key, default=None):
        """Return value for key. If not in cache or expired, return default"""
        self.lookups += 1
        try:
            pos, val, expires = self.data[key]
        except KeyError:
            self.misses += 1
            return default
        if expires > time.time():
            self.hits += 1
            self.clock_refs[pos] = True
            return val
        else:
            self.misses += 1
            self.clock_refs[pos] = False
            return default

    def put(self, key, val, timeout=None):
        """Add key to the cache with value val

        key will expire in $timeout seconds. If key is already in cache, val
        and timeout will be updated.
        """
        maxpos = self.maxpos
        clock_refs = self.clock_refs
        clock_keys = self.clock_keys
        data = self.data
        lock = self.lock
        if timeout is None:
            timeout = self.default_timeout
        with self.lock:
            entry = data.get(key)
            if entry is not None:
                pos = entry[0]
                data[key] = (pos, val, time.time() + timeout)
                clock_refs[pos] = True
                return
            hand = self.hand
            count = 0
            max_count = 107
            while 1:
                ref = clock_refs[hand]
                if ref == True:
                    clock_refs[hand] = False
                    hand += 1
                    if hand > maxpos:
                        hand = 0
                    count += 1
                    if count >= max_count:
                        clock_refs[hand] = False
                else:
                    oldkey = clock_keys[hand]
                    oldentry = data.pop(oldkey, _MARKER)
                    if oldentry is not _MARKER:
                        self.evictions += 1
                    clock_keys[hand] = key
                    clock_refs[hand] = True
                    data[key] = (hand, val, time.time() + timeout)
                    hand += 1
                    if hand > maxpos:
                        hand = 0
                    self.hand = hand
                    break

    def invalidate(self, key):
        """Remove key from the cache"""
        entry = self.data.pop(key, _MARKER)
        if entry is not _MARKER:
            self.clock_refs[entry[0]] = False