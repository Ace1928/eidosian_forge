from __future__ import annotations
import datetime
from pydantic import BaseModel
from pydantic.alias_generators import to_snake
from dataclasses import dataclass
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import MappedAsDataclass
from sqlalchemy.orm import Mapped
from sqlalchemy import MetaData
from sqlalchemy.orm import mapped_column
from sqlalchemy import Text, Table
from sqlalchemy import func as sql_func
from sqlalchemy.orm import defer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import text, select, Select, ColumnElement, and_, update, Update, delete, or_
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import registry
from ...types import errors
from typing import Optional, Type, TypeVar, Union, Set, Any, Tuple, List, Dict, cast, Generic, Generator, Callable, TYPE_CHECKING
class BasePydantic(MappedAsDataclass, DeclarativeBase, dataclass_callable=dataclass):
    """
    Base Pydantic ORM Object
    """
    if TYPE_CHECKING:
        id: Mapped[Any]
        created_at: Mapped[datetime.datetime]
        updated_at: Mapped[datetime.datetime]

    def __init__(self, **kw):
        mapper = inspect(self).mapper
        for key in mapper.relationships:
            if key in kw:
                kw[key] = mapper.relationships[key].entity.class_(**kw[key])
        super().__init__(**kw)

    @property
    def orm_class_name(self) -> str:
        """
        Returns the ORM class name
        """
        return self.__class__.__name__

    @property
    def orm_parent_class_name(self) -> str:
        """
        Returns the ORM parent class name
        """
        return self.__class__.__qualname__.split('.')[3]

    def get_non_relationship_fields(self, include: Optional[Set[str]]=None, exclude: Optional[Set[str]]=None, **kwargs) -> List[str]:
        """
        Returns the non relationship fields
        """
        include = include or set()
        exclude = exclude or set()
        return [k for k in self.__dict__ if k not in self.__mapper__.relationships or k in include or k not in exclude]

    def get_relationship_fields(self, include: Optional[Set[str]]=None, exclude: Optional[Set[str]]=None, **kwargs) -> List[str]:
        """
        Returns the relationship fields
        """
        include = include or set()
        exclude = exclude or set()
        return [k for k in self.__mapper__.relationships if k not in include or k in exclude]

    def get_exportable_kwargs(self, include: Any=None, exclude: Any=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, **kwargs) -> Dict[str, Any]:
        """
        Returns the exportable kwargs
        """
        data = {k: v for k, v in self.__dict__.items() if k in self.get_non_relationship_fields(include=include, exclude=exclude, **kwargs)}
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        if exclude_unset or exclude_defaults:
            data = {k: v for k, v in data.items() if v != self.__mapper__.columns[k].default.arg}
        return data