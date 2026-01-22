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
def _prep_for_joins(self, left_alias, subq_path):
    to_join = []
    pairs = list(subq_path.pairs())
    for i, (mapper, prop) in enumerate(pairs):
        if i > 0:
            prev_mapper = pairs[i - 1][1].mapper
            to_append = prev_mapper if prev_mapper.isa(mapper) else mapper
        else:
            to_append = mapper
        to_join.append((to_append, prop.key))
    if len(to_join) < 2:
        parent_alias = left_alias
    else:
        info = inspect(to_join[-1][0])
        if info.is_aliased_class:
            parent_alias = info.entity
        else:
            parent_alias = orm_util.AliasedClass(info.entity, use_mapper_path=True)
    local_cols = self.parent_property.local_columns
    local_attr = [getattr(parent_alias, self.parent._columntoproperty[c].key) for c in local_cols]
    return (to_join, local_attr, parent_alias)