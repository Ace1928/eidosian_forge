from __future__ import annotations
import pickle
import threading
import types
from datetime import timedelta
from typing import Any, Callable, Final, Literal, TypeVar, Union, cast, overload
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import runtime
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import CacheError, CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import (
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.caching.storage import (
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.dummy_cache_storage import (
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.time_util import time_to_seconds
class DataCaches(CacheStatsProvider):
    """Manages all DataCache instances"""

    def __init__(self):
        self._caches_lock = threading.Lock()
        self._function_caches: dict[str, DataCache] = {}

    def get_cache(self, key: str, persist: CachePersistType, max_entries: int | None, ttl: int | float | timedelta | str | None, display_name: str, allow_widgets: bool) -> DataCache:
        """Return the mem cache for the given key.

        If it doesn't exist, create a new one with the given params.
        """
        ttl_seconds = time_to_seconds(ttl, coerce_none_to_inf=False)
        with self._caches_lock:
            cache = self._function_caches.get(key)
            if cache is not None and cache.ttl_seconds == ttl_seconds and (cache.max_entries == max_entries) and (cache.persist == persist):
                return cache
            if cache is not None:
                _LOGGER.debug('Closing existing DataCache storage (key=%s, persist=%s, max_entries=%s, ttl=%s) before creating new one with different params', key, persist, max_entries, ttl)
                cache.storage.close()
            _LOGGER.debug('Creating new DataCache (key=%s, persist=%s, max_entries=%s, ttl=%s)', key, persist, max_entries, ttl)
            cache_context = self.create_cache_storage_context(function_key=key, function_name=display_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)
            cache_storage_manager = self.get_storage_manager()
            storage = cache_storage_manager.create(cache_context)
            cache = DataCache(key=key, storage=storage, persist=persist, max_entries=max_entries, ttl_seconds=ttl_seconds, display_name=display_name, allow_widgets=allow_widgets)
            self._function_caches[key] = cache
            return cache

    def clear_all(self) -> None:
        """Clear all in-memory and on-disk caches."""
        with self._caches_lock:
            try:
                self.get_storage_manager().clear_all()
            except NotImplementedError:
                for data_cache in self._function_caches.values():
                    data_cache.clear()
                    data_cache.storage.close()
            self._function_caches = {}

    def get_stats(self) -> list[CacheStat]:
        with self._caches_lock:
            function_caches = self._function_caches.copy()
        stats: list[CacheStat] = []
        for cache in function_caches.values():
            stats.extend(cache.get_stats())
        return group_stats(stats)

    def validate_cache_params(self, function_name: str, persist: CachePersistType, max_entries: int | None, ttl: int | float | timedelta | str | None) -> None:
        """Validate that the cache params are valid for given storage.

        Raises
        ------
        InvalidCacheStorageContext
            Raised if the cache storage manager is not able to work with provided
            CacheStorageContext.
        """
        ttl_seconds = time_to_seconds(ttl, coerce_none_to_inf=False)
        cache_context = self.create_cache_storage_context(function_key='DUMMY_KEY', function_name=function_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)
        try:
            self.get_storage_manager().check_context(cache_context)
        except InvalidCacheStorageContext as e:
            _LOGGER.error('Cache params for function %s are incompatible with current cache storage manager: %s', function_name, e)
            raise

    def create_cache_storage_context(self, function_key: str, function_name: str, persist: CachePersistType, ttl_seconds: float | None, max_entries: int | None) -> CacheStorageContext:
        return CacheStorageContext(function_key=function_key, function_display_name=function_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)

    def get_storage_manager(self) -> CacheStorageManager:
        if runtime.exists():
            return runtime.get_instance().cache_storage_manager
        else:
            _LOGGER.warning('No runtime found, using MemoryCacheStorageManager')
            return MemoryCacheStorageManager()