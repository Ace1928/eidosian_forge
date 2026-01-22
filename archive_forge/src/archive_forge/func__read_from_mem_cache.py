from __future__ import annotations
import math
import threading
from cachetools import TTLCache
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.stats import CacheStat
def _read_from_mem_cache(self, key: str) -> bytes:
    with self._mem_cache_lock:
        if key in self._mem_cache:
            entry = bytes(self._mem_cache[key])
            _LOGGER.debug('Memory cache HIT: %s', key)
            return entry
        else:
            _LOGGER.debug('Memory cache MISS: %s', key)
            raise CacheStorageKeyNotFoundError('Key not found in mem cache')