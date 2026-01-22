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
@log.class_logger
@properties.ColumnProperty.strategy_for(instrument=True, deferred=False)
class ColumnLoader(LoaderStrategy):
    """Provide loading behavior for a :class:`.ColumnProperty`."""
    __slots__ = ('columns', 'is_composite')

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.columns = self.parent_property.columns
        self.is_composite = hasattr(self.parent_property, 'composite_class')

    def setup_query(self, compile_state, query_entity, path, loadopt, adapter, column_collection, memoized_populators, check_for_adapt=False, **kwargs):
        for c in self.columns:
            if adapter:
                if check_for_adapt:
                    c = adapter.adapt_check_present(c)
                    if c is None:
                        return
                else:
                    c = adapter.columns[c]
            compile_state._append_dedupe_col_collection(c, column_collection)
        fetch = self.columns[0]
        if adapter:
            fetch = adapter.columns[fetch]
            if fetch is None:
                return
        memoized_populators[self.parent_property] = fetch

    def init_class_attribute(self, mapper):
        self.is_class_level = True
        coltype = self.columns[0].type
        active_history = self.parent_property.active_history or self.columns[0].primary_key or (mapper.version_id_col is not None and mapper._columntoproperty.get(mapper.version_id_col, None) is self.parent_property)
        _register_attribute(self.parent_property, mapper, useobject=False, compare_function=coltype.compare_values, active_history=active_history)

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        for col in self.columns:
            if adapter:
                col = adapter.columns[col]
            getter = result._getter(col, False)
            if getter:
                populators['quick'].append((self.key, getter))
                break
        else:
            populators['expire'].append((self.key, True))