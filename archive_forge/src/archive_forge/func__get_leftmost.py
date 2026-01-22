from __future__ import annotations
import collections
import itertools
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import interfaces
from . import loading
from . import path_registry
from . import properties
from . import query
from . import relationships
from . import unitofwork
from . import util as orm_util
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import ATTR_WAS_SET
from .base import LoaderCallableStatus
from .base import PASSIVE_OFF
from .base import PassiveFlag
from .context import _column_descriptions
from .context import ORMCompileState
from .context import ORMSelectCompileState
from .context import QueryContext
from .interfaces import LoaderStrategy
from .interfaces import StrategizedProperty
from .session import _state_session
from .state import InstanceState
from .strategy_options import Load
from .util import _none_set
from .util import AliasedClass
from .. import event
from .. import exc as sa_exc
from .. import inspect
from .. import log
from .. import sql
from .. import util
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
def _get_leftmost(self, orig_query_entity_index, subq_path, current_compile_state, is_root):
    given_subq_path = subq_path
    subq_path = subq_path.path
    subq_mapper = orm_util._class_to_mapper(subq_path[0])
    if self.parent.isa(subq_mapper) and self.parent_property is subq_path[1]:
        leftmost_mapper, leftmost_prop = (self.parent, self.parent_property)
    else:
        leftmost_mapper, leftmost_prop = (subq_mapper, subq_path[1])
    if is_root:
        new_subq_path = current_compile_state._entities[orig_query_entity_index].entity_zero._path_registry[leftmost_prop]
        additional = len(subq_path) - len(new_subq_path)
        if additional:
            new_subq_path += path_registry.PathRegistry.coerce(subq_path[-additional:])
    else:
        new_subq_path = given_subq_path
    leftmost_cols = leftmost_prop.local_columns
    leftmost_attr = [getattr(new_subq_path.path[0].entity, leftmost_mapper._columntoproperty[c].key) for c in leftmost_cols]
    return (leftmost_mapper, leftmost_attr, leftmost_prop, new_subq_path)