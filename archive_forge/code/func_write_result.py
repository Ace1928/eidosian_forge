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