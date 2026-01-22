from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
@final
class ReflectionDefaults:
    """provides blank default values for reflection methods."""

    @classmethod
    def columns(cls) -> List[ReflectedColumn]:
        return []

    @classmethod
    def pk_constraint(cls) -> ReflectedPrimaryKeyConstraint:
        return {'name': None, 'constrained_columns': []}

    @classmethod
    def foreign_keys(cls) -> List[ReflectedForeignKeyConstraint]:
        return []

    @classmethod
    def indexes(cls) -> List[ReflectedIndex]:
        return []

    @classmethod
    def unique_constraints(cls) -> List[ReflectedUniqueConstraint]:
        return []

    @classmethod
    def check_constraints(cls) -> List[ReflectedCheckConstraint]:
        return []

    @classmethod
    def table_options(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def table_comment(cls) -> ReflectedTableComment:
        return {'text': None}