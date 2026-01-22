from __future__ import annotations
import math
import threading
import types
from datetime import timedelta
from typing import Any, Callable, Final, TypeVar, cast, overload
from cachetools import TTLCache
from typing_extensions import TypeAlias
import streamlit as st
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.cache_errors import CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.time_util import time_to_seconds
class ResourceCaches(CacheStatsProvider):
    """Manages all ResourceCache instances"""

    def __init__(self):
        self._caches_lock = threading.Lock()
        self._function_caches: dict[str, ResourceCache] = {}

    def get_cache(self, key: str, display_name: str, max_entries: int | float | None, ttl: float | timedelta | str | None, validate: ValidateFunc | None, allow_widgets: bool) -> ResourceCache:
        """Return the mem cache for the given key.

        If it doesn't exist, create a new one with the given params.
        """
        if max_entries is None:
            max_entries = math.inf
        ttl_seconds = time_to_seconds(ttl)
        with self._caches_lock:
            cache = self._function_caches.get(key)
            if cache is not None and cache.ttl_seconds == ttl_seconds and (cache.max_entries == max_entries) and _equal_validate_funcs(cache.validate, validate):
                return cache
            _LOGGER.debug('Creating new ResourceCache (key=%s)', key)
            cache = ResourceCache(key=key, display_name=display_name, max_entries=max_entries, ttl_seconds=ttl_seconds, validate=validate, allow_widgets=allow_widgets)
            self._function_caches[key] = cache
            return cache

    def clear_all(self) -> None:
        """Clear all resource caches."""
        with self._caches_lock:
            self._function_caches = {}

    def get_stats(self) -> list[CacheStat]:
        with self._caches_lock:
            function_caches = self._function_caches.copy()
        stats: list[CacheStat] = []
        for cache in function_caches.values():
            stats.extend(cache.get_stats())
        return group_stats(stats)