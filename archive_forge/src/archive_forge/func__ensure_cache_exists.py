from __future__ import annotations
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from langchain_core.utils import get_from_env
def _ensure_cache_exists(cache_client: momento.CacheClient, cache_name: str) -> None:
    """Create cache if it doesn't exist.

    Raises:
        SdkException: Momento service or network error
        Exception: Unexpected response
    """
    from momento.responses import CreateCache
    create_cache_response = cache_client.create_cache(cache_name)
    if isinstance(create_cache_response, CreateCache.Success) or isinstance(create_cache_response, CreateCache.CacheAlreadyExists):
        return None
    elif isinstance(create_cache_response, CreateCache.Error):
        raise create_cache_response.inner_exception
    else:
        raise Exception(f'Unexpected response cache creation: {create_cache_response}')