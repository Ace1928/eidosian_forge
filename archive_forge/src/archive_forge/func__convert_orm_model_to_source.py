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
def _convert_orm_model_to_source(self, obj: ObjectResult, source_model: Type[SourceSchemaType], **kwargs) -> SourceSchemaType:
    """
        Converts the ORM model to the source model
        """
    values: Dict[str, Union[Dict[str, Any], Any]] = obj.model_dump(**kwargs)
    field_names, field_aliases = ([], {})
    for name, field in source_model.model_fields.items():
        field_names.append(name)
        if field.alias:
            field_aliases[field.alias] = name
    if not isinstance(values, dict):
        values = values.__dict__
        _ = values.pop('_sa_instance_state', None)
    if 'metadata_' in values and 'metadata' not in values:
        values['metadata'] = values.pop('metadata_')
    value_keys = set(values.keys())
    metadata = {key: values.pop(key) for key in value_keys if key not in field_names and key not in field_aliases}
    _ = metadata.pop('_sa_instance_state', None)
    if not values.get('metadata'):
        values['metadata'] = metadata
    else:
        metadata = {k: v for k, v in metadata.items() if v is not None}
        values['metadata'].update(metadata)
    for key, value in values['metadata'].items():
        if value is None:
            continue
        if key in field_names:
            values[key] = value
            continue
        if key in field_aliases:
            values[field_aliases[key]] = value
            continue
    return source_model.model_validate(values)