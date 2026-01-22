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
class SQLAlchemyMd5Cache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Type[FullMd5LLMCache]=FullMd5LLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        rows = self._search_rows(prompt, llm_string)
        if rows:
            return [loads(row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        with Session(self.engine) as session, session.begin():
            self._delete_previous(session, prompt, llm_string)
            prompt_md5 = self.get_md5(prompt)
            items = [self.cache_schema(id=str(uuid.uuid1()), prompt=prompt, prompt_md5=prompt_md5, llm=llm_string, response=dumps(gen), idx=i) for i, gen in enumerate(return_val)]
            for item in items:
                session.merge(item)

    def _delete_previous(self, session: Session, prompt: str, llm_string: str) -> None:
        stmt = delete(self.cache_schema).where(self.cache_schema.prompt_md5 == self.get_md5(prompt)).where(self.cache_schema.llm == llm_string).where(self.cache_schema.prompt == prompt)
        session.execute(stmt)

    def _search_rows(self, prompt: str, llm_string: str) -> Sequence[Row]:
        prompt_pd5 = self.get_md5(prompt)
        stmt = select(self.cache_schema.response).where(self.cache_schema.prompt_md5 == prompt_pd5).where(self.cache_schema.llm == llm_string).where(self.cache_schema.prompt == prompt).order_by(self.cache_schema.idx)
        with Session(self.engine) as session:
            return session.execute(stmt).fetchall()

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        with Session(self.engine) as session:
            session.execute(self.cache_schema.delete())

    @staticmethod
    def get_md5(input_string: str) -> str:
        return hashlib.md5(input_string.encode()).hexdigest()