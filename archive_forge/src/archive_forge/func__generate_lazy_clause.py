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
def _generate_lazy_clause(self, state, passive):
    criterion, param_keys = self._simple_lazy_clause
    if state is None:
        return sql_util.adapt_criterion_to_null(criterion, [key for key, ident, value in param_keys])
    mapper = self.parent_property.parent
    o = state.obj()
    dict_ = attributes.instance_dict(o)
    if passive & PassiveFlag.INIT_OK:
        passive ^= PassiveFlag.INIT_OK
    params = {}
    for key, ident, value in param_keys:
        if ident is not None:
            if passive and passive & PassiveFlag.LOAD_AGAINST_COMMITTED:
                value = mapper._get_committed_state_attr_by_column(state, dict_, ident, passive)
            else:
                value = mapper._get_state_attr_by_column(state, dict_, ident, passive)
        params[key] = value
    return (criterion, params)