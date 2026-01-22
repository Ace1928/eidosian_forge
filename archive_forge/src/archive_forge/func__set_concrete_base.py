from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
def _set_concrete_base(self, mapper):
    """Set the given :class:`_orm.Mapper` as the 'inherits' for this
        :class:`_orm.Mapper`, assuming this :class:`_orm.Mapper` is concrete
        and does not already have an inherits."""
    assert self.concrete
    assert not self.inherits
    assert isinstance(mapper, Mapper)
    self.inherits = mapper
    self.inherits.polymorphic_map.update(self.polymorphic_map)
    self.polymorphic_map = self.inherits.polymorphic_map
    for mapper in self.iterate_to_root():
        if mapper.polymorphic_on is not None:
            mapper._requires_row_aliasing = True
    self.batch = self.inherits.batch
    for mp in self.self_and_descendants:
        mp.base_mapper = self.inherits.base_mapper
    self.inherits._inheriting_mappers.append(self)
    self.passive_updates = self.inherits.passive_updates
    self._all_tables = self.inherits._all_tables
    for key, prop in mapper._props.items():
        if key not in self._props and (not self._should_exclude(key, key, local=False, column=None)):
            self._adapt_inherited_property(key, prop, False)