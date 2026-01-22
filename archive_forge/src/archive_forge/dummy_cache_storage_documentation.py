from __future__ import annotations
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (

        Dummy gets the value for a given key,
        always raises an CacheStorageKeyNotFoundError
        