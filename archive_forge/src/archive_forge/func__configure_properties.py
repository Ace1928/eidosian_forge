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
def _configure_properties(self) -> None:
    self.columns = self.c = sql_base.ColumnCollection()
    self._props = util.OrderedDict()
    self._columntoproperty = _ColumnMapping(self)
    explicit_col_props_by_column: Dict[KeyedColumnElement[Any], Tuple[str, ColumnProperty[Any]]] = {}
    explicit_col_props_by_key: Dict[str, ColumnProperty[Any]] = {}
    if self._init_properties:
        for key, prop_arg in self._init_properties.items():
            if not isinstance(prop_arg, MapperProperty):
                possible_col_prop = self._make_prop_from_column(key, prop_arg)
            else:
                possible_col_prop = prop_arg
            _map_as_property_now = True
            if isinstance(possible_col_prop, properties.ColumnProperty):
                for given_col in possible_col_prop.columns:
                    if self.local_table.c.contains_column(given_col):
                        _map_as_property_now = False
                        explicit_col_props_by_key[key] = possible_col_prop
                        explicit_col_props_by_column[given_col] = (key, possible_col_prop)
            if _map_as_property_now:
                self._configure_property(key, possible_col_prop, init=False)
    if self.inherits:
        for key, inherited_prop in self.inherits._props.items():
            if self._should_exclude(key, key, local=False, column=None):
                continue
            incoming_prop = explicit_col_props_by_key.get(key)
            if incoming_prop:
                new_prop = self._reconcile_prop_with_incoming_columns(key, inherited_prop, warn_only=False, incoming_prop=incoming_prop)
                explicit_col_props_by_key[key] = new_prop
                for inc_col in incoming_prop.columns:
                    explicit_col_props_by_column[inc_col] = (key, new_prop)
            elif key not in self._props:
                self._adapt_inherited_property(key, inherited_prop, False)
    for column in self.persist_selectable.columns:
        if column in explicit_col_props_by_column:
            key, prop = explicit_col_props_by_column[column]
            self._configure_property(key, prop, init=False)
            continue
        elif column in self._columntoproperty:
            continue
        column_key = (self.column_prefix or '') + column.key
        if self._should_exclude(column.key, column_key, local=self.local_table.c.contains_column(column), column=column):
            continue
        for mapper in self.iterate_to_root():
            if column in mapper._columntoproperty:
                column_key = mapper._columntoproperty[column].key
        self._configure_property(column_key, column, init=False, setparent=True)