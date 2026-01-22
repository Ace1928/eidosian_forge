from __future__ import annotations
import gc
import abc
import asyncio
import datetime
import contextlib
from pathlib import Path
from pydantic.networks import PostgresDsn
from pydantic_settings import BaseSettings
from pydantic import validator, model_validator, computed_field, BaseModel, Field, PrivateAttr
from sqlalchemy import text as sql_text, TextClause
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from lazyops.utils.logs import logger, Logger
from lazyops.utils.lazy import lazy_import
from ...utils.helpers import update_dict
from typing import Any, Dict, List, Optional, Type, Literal, Iterable, Tuple, TypeVar, Union, Annotated, Callable, Generator, AsyncGenerator, Set, TYPE_CHECKING
class RemoteDatabaseConnection(PostgresConfig):
    """
    Remote Database
    """
    name: str
    extra_kws: Optional[Dict[str, Any]] = Field(default_factory=dict)
    _engine_rw: Optional[AsyncEngine] = PrivateAttr(None)
    _engine_ro: Optional[AsyncEngine] = PrivateAttr(None)
    _session_rw: Optional[async_sessionmaker[AsyncSession]] = PrivateAttr(None)
    _session_ro: Optional[async_sessionmaker[AsyncSession]] = PrivateAttr(None)

    @property
    def engine(self) -> AsyncEngine:
        """
        Returns the engine
        """
        if self._engine_rw is None:
            self._engine_rw = create_async_engine(**self.get_engine_kwargs(readonly=False, **self.extra_kws))
        return self._engine_rw

    @property
    def engine_ro(self) -> AsyncEngine:
        """
        Returns the readonly engine
        """
        if self._engine_ro is None:
            self._engine_ro = create_async_engine(**self.get_engine_kwargs(readonly=True, **self.extra_kws))
        return self._engine_ro

    @property
    def session_rw(self) -> async_sessionmaker[AsyncSession]:
        """
        Returns the session
        """
        if self._session_rw is None:
            self._session_rw = async_sessionmaker(self.engine, class_=AsyncSession, **self.get_session_kwargs(readonly=False, **self.extra_kws))
        return self._session_rw

    @property
    def session_ro(self) -> async_sessionmaker[AsyncSession]:
        """
        Returns the readonly session
        """
        if self._session_ro is None:
            self._session_ro = async_sessionmaker(self.engine_ro, class_=AsyncSession, **self.get_session_kwargs(readonly=True, **self.extra_kws))
        return self._session_ro