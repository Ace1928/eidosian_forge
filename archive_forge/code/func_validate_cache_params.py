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