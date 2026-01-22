from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
class lru_cache(object):
    """ Decorator for LRU-cached function

    timeout parameter specifies after how many seconds a cached entry should
    be considered invalid.
    """

    def __init__(self, maxsize, cache=None, timeout=None, ignore_unhashable_args=False):
        if cache is None:
            if maxsize is None:
                cache = UnboundedCache()
            elif timeout is None:
                cache = LRUCache(maxsize)
            else:
                cache = ExpiringLRUCache(maxsize, default_timeout=timeout)
        self.cache = cache
        self._ignore_unhashable_args = ignore_unhashable_args

    def __call__(self, func):
        cache = self.cache
        marker = _MARKER

        def cached_wrapper(*args, **kwargs):
            try:
                key = (args, frozenset(kwargs.items())) if kwargs else args
            except TypeError as e:
                if self._ignore_unhashable_args:
                    return func(*args, **kwargs)
                else:
                    raise e
            else:
                val = cache.get(key, marker)
                if val is marker:
                    val = func(*args, **kwargs)
                    cache.put(key, val)
                return val

        def _maybe_copy(source, target, attr):
            value = getattr(source, attr, source)
            if value is not source:
                setattr(target, attr, value)
        _maybe_copy(func, cached_wrapper, '__module__')
        _maybe_copy(func, cached_wrapper, '__name__')
        _maybe_copy(func, cached_wrapper, '__doc__')
        cached_wrapper._cache = cache
        return cached_wrapper