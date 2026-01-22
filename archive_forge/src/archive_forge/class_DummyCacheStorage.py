from __future__ import annotations
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (
class DummyCacheStorage(CacheStorage):

    def get(self, key: str) -> bytes:
        """
        Dummy gets the value for a given key,
        always raises an CacheStorageKeyNotFoundError
        """
        raise CacheStorageKeyNotFoundError('Key not found in dummy cache')

    def set(self, key: str, value: bytes) -> None:
        pass

    def delete(self, key: str) -> None:
        pass

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass