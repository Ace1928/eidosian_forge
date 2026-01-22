from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol
@dataclass(frozen=True)
class CacheStorageContext:
    """Context passed to the cache storage during initialization
    This is the normalized parameters that are passed to CacheStorageManager.create()
    method.

    Parameters
    ----------
    function_key: str
        A hash computed based on function name and source code decorated
        by `@st.cache_data`

    function_display_name: str
        The display name of the function that is decorated by `@st.cache_data`

    ttl_seconds : float or None
        The time-to-live for the keys in storage, in seconds. If None, the entry
        will never expire.

    max_entries : int or None
        The maximum number of entries to store in the cache storage.
        If None, the cache storage will not limit the number of entries.

    persist : Literal["disk"] or None
        The persistence mode for the cache storage.
        Legacy parameter, that used in Streamlit current cache storage implementation.
        Could be ignored by cache storage implementation, if storage does not support
        persistence or it persistent by default.
    """
    function_key: str
    function_display_name: str
    ttl_seconds: float | None = None
    max_entries: int | None = None
    persist: Literal['disk'] | None = None