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
class DataCache(Cache):
    """Manages cached values for a single st.cache_data function."""

    def __init__(self, key: str, storage: CacheStorage, persist: CachePersistType, max_entries: int | None, ttl_seconds: float | None, display_name: str, allow_widgets: bool=False):
        super().__init__()
        self.key = key
        self.display_name = display_name
        self.storage = storage
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.persist = persist
        self.allow_widgets = allow_widgets

    def get_stats(self) -> list[CacheStat]:
        if isinstance(self.storage, CacheStatsProvider):
            return self.storage.get_stats()
        return []

    def read_result(self, key: str) -> CachedResult:
        """Read a value and messages from the cache. Raise `CacheKeyNotFoundError`
        if the value doesn't exist, and `CacheError` if the value exists but can't
        be unpickled.
        """
        try:
            pickled_entry = self.storage.get(key)
        except CacheStorageKeyNotFoundError as e:
            raise CacheKeyNotFoundError(str(e)) from e
        except CacheStorageError as e:
            raise CacheError(str(e)) from e
        try:
            entry = pickle.loads(pickled_entry)
            if not isinstance(entry, MultiCacheResults):
                self.storage.delete(key)
                raise CacheKeyNotFoundError()
            ctx = get_script_run_ctx()
            if not ctx:
                raise CacheKeyNotFoundError()
            widget_key = entry.get_current_widget_key(ctx, CacheType.DATA)
            if widget_key in entry.results:
                return entry.results[widget_key]
            else:
                raise CacheKeyNotFoundError()
        except pickle.UnpicklingError as exc:
            raise CacheError(f'Failed to unpickle {key}') from exc

    @gather_metrics('_cache_data_object')
    def write_result(self, key: str, value: Any, messages: list[MsgData]) -> None:
        """Write a value and associated messages to the cache.
        The value must be pickleable.
        """
        ctx = get_script_run_ctx()
        if ctx is None:
            return
        main_id = st._main.id
        sidebar_id = st.sidebar.id
        if self.allow_widgets:
            widgets = {msg.widget_metadata.widget_id for msg in messages if isinstance(msg, ElementMsgData) and msg.widget_metadata is not None}
        else:
            widgets = set()
        multi_cache_results: MultiCacheResults | None = None
        try:
            multi_cache_results = self._read_multi_results_from_storage(key)
        except (CacheKeyNotFoundError, pickle.UnpicklingError):
            pass
        if multi_cache_results is None:
            multi_cache_results = MultiCacheResults(widget_ids=widgets, results={})
        multi_cache_results.widget_ids.update(widgets)
        widget_key = multi_cache_results.get_current_widget_key(ctx, CacheType.DATA)
        result = CachedResult(value, messages, main_id, sidebar_id)
        multi_cache_results.results[widget_key] = result
        try:
            pickled_entry = pickle.dumps(multi_cache_results)
        except (pickle.PicklingError, TypeError) as exc:
            raise CacheError(f'Failed to pickle {key}') from exc
        self.storage.set(key, pickled_entry)

    def _clear(self) -> None:
        self.storage.clear()

    def _read_multi_results_from_storage(self, key: str) -> MultiCacheResults:
        """Look up the results from storage and ensure it has the right type.

        Raises a `CacheKeyNotFoundError` if the key has no entry, or if the
        entry is malformed.
        """
        try:
            pickled = self.storage.get(key)
        except CacheStorageKeyNotFoundError as e:
            raise CacheKeyNotFoundError(str(e)) from e
        maybe_results = pickle.loads(pickled)
        if isinstance(maybe_results, MultiCacheResults):
            return maybe_results
        else:
            self.storage.delete(key)
            raise CacheKeyNotFoundError()