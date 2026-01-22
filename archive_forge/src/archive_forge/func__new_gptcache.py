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