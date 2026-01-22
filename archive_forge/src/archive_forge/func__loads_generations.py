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
def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """
    Deserialization of a string into a generic RETURN_VAL_TYPE
    (i.e. a sequence of `Generation`).

    See `_dumps_generations`, the inverse of this function.

    Args:
        generations_str (str): A string representing a list of generations.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        generations = [loads(_item_str) for _item_str in json.loads(generations_str)]
        return generations
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        gen_dicts = json.loads(generations_str)
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(f"Legacy 'Generation' cached blob encountered: '{generations_str}'")
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Malformed/unparsable cached blob encountered: '{generations_str}'")
        return None