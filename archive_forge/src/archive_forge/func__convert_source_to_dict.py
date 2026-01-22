from __future__ import annotations
from lazyops.imports._sqlalchemy import require_sql
import datetime
from uuid import UUID
from pydantic import BaseModel
from pydantic.alias_generators import to_snake
from dataclasses import dataclass
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import MappedAsDataclass
from sqlalchemy.orm import Mapped, InstrumentedAttribute
from sqlalchemy.orm import mapped_column
from sqlalchemy import func as sql_func
from sqlalchemy import Text
from sqlalchemy.orm import defer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import text, select, Select, ColumnElement, and_, update, Update, delete, or_
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import registry
from lazyops.utils.lazy import lazy_import
from lazyops.utils.logs import logger
from typing import Optional, Type, TypeVar, Union, Set, Any, Tuple, Literal, List, Dict, cast, Generic, Generator, Callable, TYPE_CHECKING
from . import errors
def _convert_source_to_dict(self, source: SourceSchemaType, model_dump_kwargs: Optional[Dict[str, Any]]=None, use_encoder: Optional[bool]=True, **kwargs) -> Dict[str, Any]:
    """
        [model_dump | jsonable_encoder] Converts the source to a dict
        """
    dump_kwargs = self.model_dump_kwargs.copy()
    if model_dump_kwargs:
        dump_kwargs.update(model_dump_kwargs)
    if kwargs:
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in dump_kwargs:
                dump_kwargs[k] = v
                continue
            if isinstance(dump_kwargs[k], set):
                dump_kwargs[k].add(v)
            elif isinstance(dump_kwargs[k], list):
                dump_kwargs[k].append(v)
            elif isinstance(dump_kwargs[k], dict):
                dump_kwargs[k].update(v)
            else:
                dump_kwargs[k] = v
    if use_encoder:
        return jsonable_encoder(source, **dump_kwargs)
    return source.model_dump(**dump_kwargs)