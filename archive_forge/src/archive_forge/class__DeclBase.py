from __future__ import annotations
from typing import Any
import sqlalchemy as sa
from .base import TestBase
from .sql import TablesTest
from .. import assertions
from .. import config
from .. import schema
from ..entities import BasicEntity
from ..entities import ComparableEntity
from ..util import adict
from ... import orm
from ...orm import DeclarativeBase
from ...orm import events as orm_events
from ...orm import registry
class _DeclBase(DeclarativeBase):
    __table_cls__ = schema.Table
    metadata = cls._tables_metadata
    type_annotation_map = {str: sa.String().with_variant(sa.String(50), 'mysql', 'mariadb', 'oracle')}

    def __init_subclass__(cls, **kw) -> None:
        assert cls_registry is not None
        cls_registry[cls.__name__] = cls
        super().__init_subclass__(**kw)