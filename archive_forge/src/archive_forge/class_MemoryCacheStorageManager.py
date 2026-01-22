from __future__ import annotations
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (
class MemoryCacheStorageManager(CacheStorageManager):

    def create(self, context: CacheStorageContext) -> CacheStorage:
        """Creates a new cache storage instance wrapped with in-memory cache layer"""
        persist_storage = DummyCacheStorage()
        return InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)

    def clear_all(self) -> None:
        raise NotImplementedError

    def check_context(self, context: CacheStorageContext) -> None:
        pass