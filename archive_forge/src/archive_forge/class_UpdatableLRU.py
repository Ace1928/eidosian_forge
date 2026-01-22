from __future__ import annotations
import collections
import functools
import logging
import math
import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
class UpdatableLRU(Generic[P, T]):
    """
    Custom implementation of LRU cache that allows updating keys

    Used by BackgroudBlockCache
    """

    class CacheInfo(NamedTuple):
        hits: int
        misses: int
        maxsize: int
        currsize: int

    def __init__(self, func: Callable[P, T], max_size: int=128) -> None:
        self._cache: OrderedDict[Any, T] = collections.OrderedDict()
        self._func = func
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if kwargs:
            raise TypeError(f'Got unexpected keyword argument {kwargs.keys()}')
        with self._lock:
            if args in self._cache:
                self._cache.move_to_end(args)
                self._hits += 1
                return self._cache[args]
        result = self._func(*args, **kwargs)
        with self._lock:
            self._cache[args] = result
            self._misses += 1
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return result

    def is_key_cached(self, *args: Any) -> bool:
        with self._lock:
            return args in self._cache

    def add_key(self, result: T, *args: Any) -> None:
        with self._lock:
            self._cache[args] = result
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def cache_info(self) -> UpdatableLRU.CacheInfo:
        with self._lock:
            return self.CacheInfo(maxsize=self._max_size, currsize=len(self._cache), hits=self._hits, misses=self._misses)