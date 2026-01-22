from __future__ import annotations
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC
from datetime import timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import (
from sqlalchemy import Column, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from langchain_community.vectorstores.azure_cosmos_db import (
from langchain_core._api.deprecation import deprecated
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM, aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils import get_from_env
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_community.vectorstores.redis import Redis as RedisVectorstore
class GPTCache(BaseCache):
    """Cache that uses GPTCache as a backend."""

    def __init__(self, init_func: Union[Callable[[Any, str], None], Callable[[Any], None], None]=None):
        """Initialize by passing in init function (default: `None`).

        Args:
            init_func (Optional[Callable[[Any], None]]): init `GPTCache` function
            (default: `None`)

        Example:
        .. code-block:: python

            # Initialize GPTCache with a custom init function
            import gptcache
            from gptcache.processor.pre import get_prompt
            from gptcache.manager.factory import get_data_manager
            from langchain_community.globals import set_llm_cache

            # Avoid multiple caches using the same file,
            causing different llm model caches to affect each other

            def init_gptcache(cache_obj: gptcache.Cache, llm str):
                cache_obj.init(
                    pre_embedding_func=get_prompt,
                    data_manager=manager_factory(
                        manager="map",
                        data_dir=f"map_cache_{llm}"
                    ),
                )

            set_llm_cache(GPTCache(init_gptcache))

        """
        try:
            import gptcache
        except ImportError:
            raise ImportError('Could not import gptcache python package. Please install it with `pip install gptcache`.')
        self.init_gptcache_func: Union[Callable[[Any, str], None], Callable[[Any], None], None] = init_func
        self.gptcache_dict: Dict[str, Any] = {}

    def _new_gptcache(self, llm_string: str) -> Any:
        """New gptcache object"""
        from gptcache import Cache
        from gptcache.manager.factory import get_data_manager
        from gptcache.processor.pre import get_prompt
        _gptcache = Cache()
        if self.init_gptcache_func is not None:
            sig = inspect.signature(self.init_gptcache_func)
            if len(sig.parameters) == 2:
                self.init_gptcache_func(_gptcache, llm_string)
            else:
                self.init_gptcache_func(_gptcache)
        else:
            _gptcache.init(pre_embedding_func=get_prompt, data_manager=get_data_manager(data_path=llm_string))
        self.gptcache_dict[llm_string] = _gptcache
        return _gptcache

    def _get_gptcache(self, llm_string: str) -> Any:
        """Get a cache object.

        When the corresponding llm model cache does not exist, it will be created."""
        _gptcache = self.gptcache_dict.get(llm_string, None)
        if not _gptcache:
            _gptcache = self._new_gptcache(llm_string)
        return _gptcache

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the cache data.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then retrieve the data from the cache based on the `prompt`.
        """
        from gptcache.adapter.api import get
        _gptcache = self._get_gptcache(llm_string)
        res = get(prompt, cache_obj=_gptcache)
        return _loads_generations(res) if res is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then store the `prompt` and `return_val` in the cache object.
        """
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(f'GPTCache only supports caching of normal LLM generations, got {type(gen)}')
        from gptcache.adapter.api import put
        _gptcache = self._get_gptcache(llm_string)
        handled_data = _dumps_generations(return_val)
        put(prompt, handled_data, cache_obj=_gptcache)
        return None

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        from gptcache import Cache
        for gptcache_instance in self.gptcache_dict.values():
            gptcache_instance = cast(Cache, gptcache_instance)
            gptcache_instance.flush()
        self.gptcache_dict.clear()