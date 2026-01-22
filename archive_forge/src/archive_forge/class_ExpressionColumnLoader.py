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
@properties.ColumnProperty.strategy_for(query_expression=True)
class ExpressionColumnLoader(ColumnLoader):

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        null = sql.null().label(None)
        self._have_default_expression = any((not c.compare(null) for c in self.parent_property.columns))

    def setup_query(self, compile_state, query_entity, path, loadopt, adapter, column_collection, memoized_populators, **kwargs):
        columns = None
        if loadopt and loadopt._extra_criteria:
            columns = loadopt._extra_criteria
        elif self._have_default_expression:
            columns = self.parent_property.columns
        if columns is None:
            return
        for c in columns:
            if adapter:
                c = adapter.columns[c]
            compile_state._append_dedupe_col_collection(c, column_collection)
        fetch = columns[0]
        if adapter:
            fetch = adapter.columns[fetch]
            if fetch is None:
                return
        memoized_populators[self.parent_property] = fetch

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        if loadopt and loadopt._extra_criteria:
            columns = loadopt._extra_criteria
            for col in columns:
                if adapter:
                    col = adapter.columns[col]
                getter = result._getter(col, False)
                if getter:
                    populators['quick'].append((self.key, getter))
                    break
            else:
                populators['expire'].append((self.key, True))

    def init_class_attribute(self, mapper):
        self.is_class_level = True
        _register_attribute(self.parent_property, mapper, useobject=False, compare_function=self.columns[0].type.compare_values, accepts_scalar_loader=False)