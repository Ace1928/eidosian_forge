import os
import time
import asyncio
import contextlib
from lazyops.imports._sqlalchemy import require_sql
from sqlalchemy import create_engine, event, exc
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session, scoped_session
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_scoped_session
from lazyops.utils.logs import logger
from lazyops.utils import Json
from lazyops.types import BaseModel, lazyproperty, BaseSettings, Field
from typing import Any, Generator, AsyncGenerator, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from pydantic.networks import PostgresDsn
from lazyops.libs.psqldb.retry import reconnecting_engine
from lazyops.utils.helpers import import_string
def uri_builder(uri: Union[str, SafePostgresDsn], scheme: Optional[str]=None) -> SafePostgresDsn:
    """
    Helper to construct a PostgresDsn from a string
    """
    if uri is None:
        raise ValueError('uri cannot be empty')
    if isinstance(uri, str):
        if scheme and '://' not in uri:
            uri = f'{scheme}://{uri}'
        x = Dummy(dsn=uri)
        uri = x.dsn
    if scheme and uri.scheme != scheme:
        new = uri.replace(uri.scheme, scheme)
        x = Dummy(dsn=new)
        uri = x.dsn
    return uri