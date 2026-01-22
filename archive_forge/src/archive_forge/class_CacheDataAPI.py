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
class CacheDataAPI:
    """Implements the public st.cache_data API: the @st.cache_data decorator, and
    st.cache_data.clear().
    """

    def __init__(self, decorator_metric_name: str, deprecation_warning: str | None=None):
        """Create a CacheDataAPI instance.

        Parameters
        ----------
        decorator_metric_name
            The metric name to record for decorator usage. `@st.experimental_memo` is
            deprecated, but we're still supporting it and tracking its usage separately
            from `@st.cache_data`.

        deprecation_warning
            An optional deprecation warning to show when the API is accessed.
        """
        self._decorator = gather_metrics(decorator_metric_name, self._decorator)
        self._deprecation_warning = deprecation_warning
    F = TypeVar('F', bound=Callable[..., Any])

    @overload
    def __call__(self, func: F) -> F:
        ...

    @overload
    def __call__(self, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, persist: CachePersistType | bool=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None) -> Callable[[F], F]:
        ...

    def __call__(self, func: F | None=None, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, persist: CachePersistType | bool=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None):
        return self._decorator(func, ttl=ttl, max_entries=max_entries, persist=persist, show_spinner=show_spinner, experimental_allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs)

    def _decorator(self, func: F | None=None, *, ttl: float | timedelta | str | None, max_entries: int | None, show_spinner: bool | str, persist: CachePersistType | bool, experimental_allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        """Decorator to cache functions that return data (e.g. dataframe transforms, database queries, ML inference).

        Cached objects are stored in "pickled" form, which means that the return
        value of a cached function must be pickleable. Each caller of the cached
        function gets its own copy of the cached data.

        You can clear a function's cache with ``func.clear()`` or clear the entire
        cache with ``st.cache_data.clear()``.

        To cache global resources, use ``st.cache_resource`` instead. Learn more
        about caching at https://docs.streamlit.io/library/advanced-features/caching.

        Parameters
        ----------
        func : callable
            The function to cache. Streamlit hashes the function's source code.

        ttl : float, timedelta, str, or None
            The maximum time to keep an entry in the cache. Can be one of:

            * ``None`` if cache entries should never expire (default).
            * A number specifying the time in seconds.
            * A string specifying the time in a format supported by `Pandas's
              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,
              e.g. ``"1d"``, ``"1.5 days"``, or ``"1h23s"``.
            * A ``timedelta`` object from `Python's built-in datetime library
              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,
              e.g. ``timedelta(days=1)``.

            Note that ``ttl`` will be ignored if ``persist="disk"`` or ``persist=True``.

        max_entries : int or None
            The maximum number of entries to keep in the cache, or None
            for an unbounded cache. When a new entry is added to a full cache,
            the oldest cached entry will be removed. Defaults to None.

        show_spinner : bool or str
            Enable the spinner. Default is True to show a spinner when there is
            a "cache miss" and the cached data is being created. If string,
            value of show_spinner param will be used for spinner text.

        persist : "disk", bool, or None
            Optional location to persist cached data to. Passing "disk" (or True)
            will persist the cached data to the local disk. None (or False) will disable
            persistence. The default is None.

        experimental_allow_widgets : bool
            Allow widgets to be used in the cached function. Defaults to False.
            Support for widgets in cached functions is currently experimental.
            Setting this parameter to True may lead to excessive memory use since the
            widget value is treated as an additional input parameter to the cache.
            We may remove support for this option at any time without notice.

        hash_funcs : dict or None
            Mapping of types or fully qualified names to hash functions.
            This is used to override the behavior of the hasher inside Streamlit's
            caching mechanism: when the hasher encounters an object, it will first
            check to see if its type matches a key in this dict and, if so, will use
            the provided function to generate a hash for it. See below for an example
            of how this can be used.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> @st.cache_data
        ... def fetch_and_clean_data(url):
        ...     # Fetch data from URL here, and then clean it up.
        ...     return data
        ...
        >>> d1 = fetch_and_clean_data(DATA_URL_1)
        >>> # Actually executes the function, since this is the first time it was
        >>> # encountered.
        >>>
        >>> d2 = fetch_and_clean_data(DATA_URL_1)
        >>> # Does not execute the function. Instead, returns its previously computed
        >>> # value. This means that now the data in d1 is the same as in d2.
        >>>
        >>> d3 = fetch_and_clean_data(DATA_URL_2)
        >>> # This is a different URL, so the function executes.

        To set the ``persist`` parameter, use this command as follows:

        >>> import streamlit as st
        >>>
        >>> @st.cache_data(persist="disk")
        ... def fetch_and_clean_data(url):
        ...     # Fetch data from URL here, and then clean it up.
        ...     return data

        By default, all parameters to a cached function must be hashable.
        Any parameter whose name begins with ``_`` will not be hashed. You can use
        this as an "escape hatch" for parameters that are not hashable:

        >>> import streamlit as st
        >>>
        >>> @st.cache_data
        ... def fetch_and_clean_data(_db_connection, num_rows):
        ...     # Fetch data from _db_connection here, and then clean it up.
        ...     return data
        ...
        >>> connection = make_database_connection()
        >>> d1 = fetch_and_clean_data(connection, num_rows=10)
        >>> # Actually executes the function, since this is the first time it was
        >>> # encountered.
        >>>
        >>> another_connection = make_database_connection()
        >>> d2 = fetch_and_clean_data(another_connection, num_rows=10)
        >>> # Does not execute the function. Instead, returns its previously computed
        >>> # value - even though the _database_connection parameter was different
        >>> # in both calls.

        A cached function's cache can be procedurally cleared:

        >>> import streamlit as st
        >>>
        >>> @st.cache_data
        ... def fetch_and_clean_data(_db_connection, num_rows):
        ...     # Fetch data from _db_connection here, and then clean it up.
        ...     return data
        ...
        >>> fetch_and_clean_data.clear()
        >>> # Clear all cached entries for this function.

        To override the default hashing behavior, pass a custom hash function.
        You can do that by mapping a type (e.g. ``datetime.datetime``) to a hash
        function (``lambda dt: dt.isoformat()``) like this:

        >>> import streamlit as st
        >>> import datetime
        >>>
        >>> @st.cache_data(hash_funcs={datetime.datetime: lambda dt: dt.isoformat()})
        ... def convert_to_utc(dt: datetime.datetime):
        ...     return dt.astimezone(datetime.timezone.utc)

        Alternatively, you can map the type's fully-qualified name
        (e.g. ``"datetime.datetime"``) to the hash function instead:

        >>> import streamlit as st
        >>> import datetime
        >>>
        >>> @st.cache_data(hash_funcs={"datetime.datetime": lambda dt: dt.isoformat()})
        ... def convert_to_utc(dt: datetime.datetime):
        ...     return dt.astimezone(datetime.timezone.utc)

        """
        persist_string: CachePersistType
        if persist is True:
            persist_string = 'disk'
        elif persist is False:
            persist_string = None
        else:
            persist_string = persist
        if persist_string not in (None, 'disk'):
            raise StreamlitAPIException(f"Unsupported persist option '{persist}'. Valid values are 'disk' or None.")
        self._maybe_show_deprecation_warning()

        def wrapper(f):
            return make_cached_func_wrapper(CachedDataFuncInfo(func=f, persist=persist_string, show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))
        if func is None:
            return wrapper
        return make_cached_func_wrapper(CachedDataFuncInfo(func=cast(types.FunctionType, func), persist=persist_string, show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))

    @gather_metrics('clear_data_caches')
    def clear(self) -> None:
        """Clear all in-memory and on-disk data caches."""
        self._maybe_show_deprecation_warning()
        _data_caches.clear_all()

    def _maybe_show_deprecation_warning(self):
        """If the API is being accessed with the deprecated `st.experimental_memo` name,
        show a deprecation warning.
        """
        if self._deprecation_warning is not None:
            show_deprecation_warning(self._deprecation_warning)