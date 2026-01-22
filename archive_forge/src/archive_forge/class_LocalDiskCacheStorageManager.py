from __future__ import annotations
import math
import os
import shutil
from typing import Final
from streamlit import util
from streamlit.file_util import get_streamlit_file_path, streamlit_read, streamlit_write
from streamlit.logger import get_logger
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import (
class LocalDiskCacheStorageManager(CacheStorageManager):

    def create(self, context: CacheStorageContext) -> CacheStorage:
        """Creates a new cache storage instance wrapped with in-memory cache layer"""
        persist_storage = LocalDiskCacheStorage(context)
        return InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)

    def clear_all(self) -> None:
        cache_path = get_cache_folder_path()
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)

    def check_context(self, context: CacheStorageContext) -> None:
        if context.persist == 'disk' and context.ttl_seconds is not None and (not math.isinf(context.ttl_seconds)):
            _LOGGER.warning(f"The cached function '{context.function_display_name}' has a TTL that will be ignored. Persistent cached functions currently don't support TTL.")