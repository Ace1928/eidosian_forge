from __future__ import annotations
import functools
import hashlib
import inspect
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Final
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
def _handle_cache_miss(self, cache: Cache, value_key: str, func_args: tuple[Any, ...], func_kwargs: dict[str, Any]) -> Any:
    """Handle a cache miss: compute a new cached value, write it back to the cache,
        and return that newly-computed value.
        """
    with cache.compute_value_lock(value_key):
        try:
            cached_result = cache.read_result(value_key)
            return self._handle_cache_hit(cached_result)
        except CacheKeyNotFoundError:
            pass
        with self._info.cached_message_replay_ctx.calling_cached_function(self._info.func, self._info.allow_widgets):
            computed_value = self._info.func(*func_args, **func_kwargs)
        messages = self._info.cached_message_replay_ctx._most_recent_messages
        try:
            cache.write_result(value_key, computed_value, messages)
            return computed_value
        except (CacheError, RuntimeError):
            if True in [type_util.is_type(computed_value, type_name) for type_name in UNEVALUATED_DATAFRAME_TYPES]:
                raise UnevaluatedDataFrameError(f'\n                        The function {get_cached_func_name_md(self._info.func)} is decorated with `st.cache_data` but it returns an unevaluated dataframe\n                        of type `{type_util.get_fqn_type(computed_value)}`. Please call `collect()` or `to_pandas()` on the dataframe before returning it,\n                        so `st.cache_data` can serialize and cache it.')
            raise UnserializableReturnValueError(return_value=computed_value, func=self._info.func)