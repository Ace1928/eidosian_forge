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
@util.preload_module('sqlalchemy.orm.strategy_options')
def _subclass_load_via_in(self, entity, polymorphic_from):
    """Assemble a that can load the columns local to
        this subclass as a SELECT with IN.

        """
    strategy_options = util.preloaded.orm_strategy_options
    assert self.inherits
    if self.polymorphic_on is not None:
        polymorphic_prop = self._columntoproperty[self.polymorphic_on]
        keep_props = set([polymorphic_prop] + self._identity_key_props)
    else:
        keep_props = set(self._identity_key_props)
    disable_opt = strategy_options.Load(entity)
    enable_opt = strategy_options.Load(entity)
    classes_to_include = {self}
    m: Optional[Mapper[Any]] = self.inherits
    while m is not None and m is not polymorphic_from and (m.polymorphic_load == 'selectin'):
        classes_to_include.add(m)
        m = m.inherits
    for prop in self.attrs:
        if prop.key not in self.class_manager:
            continue
        if prop.parent in classes_to_include or prop in keep_props:
            if not isinstance(prop, StrategizedProperty):
                continue
            enable_opt = enable_opt._set_generic_strategy((getattr(entity.entity_namespace, prop.key),), dict(prop.strategy_key), _reconcile_to_other=True)
        else:
            disable_opt = disable_opt._set_generic_strategy((getattr(entity.entity_namespace, prop.key),), {'do_nothing': True}, _reconcile_to_other=False)
    primary_key = [sql_util._deep_annotate(pk, {'_orm_adapt': True}) for pk in self.primary_key]
    in_expr: ColumnElement[Any]
    if len(primary_key) > 1:
        in_expr = sql.tuple_(*primary_key)
    else:
        in_expr = primary_key[0]
    if entity.is_aliased_class:
        assert entity.mapper is self
        q = sql.select(entity).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        in_expr = entity._adapter.traverse(in_expr)
        primary_key = [entity._adapter.traverse(k) for k in primary_key]
        q = q.where(in_expr.in_(sql.bindparam('primary_keys', expanding=True))).order_by(*primary_key)
    else:
        q = sql.select(self).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        q = q.where(in_expr.in_(sql.bindparam('primary_keys', expanding=True))).order_by(*primary_key)
    return (q, enable_opt, disable_opt)