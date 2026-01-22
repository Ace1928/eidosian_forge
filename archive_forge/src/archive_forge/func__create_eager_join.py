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
def _create_eager_join(self, compile_state, query_entity, path, adapter, parentmapper, clauses, innerjoin, chained_from_outerjoin, extra_criteria):
    if parentmapper is None:
        localparent = query_entity.mapper
    else:
        localparent = parentmapper
    should_nest_selectable = compile_state.multi_row_eager_loaders and compile_state._should_nest_selectable
    query_entity_key = None
    if query_entity not in compile_state.eager_joins and (not should_nest_selectable) and compile_state.from_clauses:
        indexes = sql_util.find_left_clause_that_matches_given(compile_state.from_clauses, query_entity.selectable)
        if len(indexes) > 1:
            raise sa_exc.InvalidRequestError("Can't identify which query entity in which to joined eager load from.   Please use an exact match when specifying the join path.")
        if indexes:
            clause = compile_state.from_clauses[indexes[0]]
            query_entity_key, default_towrap = (indexes[0], clause)
    if query_entity_key is None:
        query_entity_key, default_towrap = (query_entity, query_entity.selectable)
    towrap = compile_state.eager_joins.setdefault(query_entity_key, default_towrap)
    if adapter:
        if getattr(adapter, 'is_aliased_class', False):
            efm = adapter.aliased_insp._entity_for_mapper(localparent if localparent.isa(self.parent) else self.parent)
            onclause = getattr(efm.entity, self.key, self.parent_property)
        else:
            onclause = getattr(orm_util.AliasedClass(self.parent, adapter.selectable, use_mapper_path=True), self.key, self.parent_property)
    else:
        onclause = self.parent_property
    assert clauses.is_aliased_class
    attach_on_outside = not chained_from_outerjoin or not innerjoin or innerjoin == 'unnested' or query_entity.entity_zero.represents_outer_join
    extra_join_criteria = extra_criteria
    additional_entity_criteria = compile_state.global_attributes.get(('additional_entity_criteria', self.mapper), ())
    if additional_entity_criteria:
        extra_join_criteria += tuple((ae._resolve_where_criteria(self.mapper) for ae in additional_entity_criteria if ae.propagate_to_loaders))
    if attach_on_outside:
        eagerjoin = orm_util._ORMJoin(towrap, clauses.aliased_insp, onclause, isouter=not innerjoin or query_entity.entity_zero.represents_outer_join or (chained_from_outerjoin and isinstance(towrap, sql.Join)), _left_memo=self.parent, _right_memo=self.mapper, _extra_criteria=extra_join_criteria)
    else:
        eagerjoin = self._splice_nested_inner_join(path, towrap, clauses, onclause, extra_join_criteria)
    compile_state.eager_joins[query_entity_key] = eagerjoin
    eagerjoin.stop_on = query_entity.selectable
    if not parentmapper:
        for col in sql_util._find_columns(self.parent_property.primaryjoin):
            if localparent.persist_selectable.c.contains_column(col):
                if adapter:
                    col = adapter.columns[col]
                compile_state._append_dedupe_col_collection(col, compile_state.primary_columns)
    if self.parent_property.order_by:
        compile_state.eager_order_by += tuple(eagerjoin._target_adapter.copy_and_process(util.to_list(self.parent_property.order_by)))