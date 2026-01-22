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
def _generate_row_adapter(self, compile_state, entity, path, loadopt, adapter, column_collection, parentmapper, chained_from_outerjoin):
    with_poly_entity = path.get(compile_state.attributes, 'path_with_polymorphic', None)
    if with_poly_entity:
        to_adapt = with_poly_entity
    else:
        insp = inspect(self.entity)
        if insp.is_aliased_class:
            alt_selectable = insp.selectable
        else:
            alt_selectable = None
        to_adapt = orm_util.AliasedClass(self.mapper, alias=alt_selectable._anonymous_fromclause(flat=True) if alt_selectable is not None else None, flat=True, use_mapper_path=True)
    to_adapt_insp = inspect(to_adapt)
    clauses = to_adapt_insp._memo(('joinedloader_ormadapter', self), orm_util.ORMAdapter, orm_util._TraceAdaptRole.JOINEDLOAD_MEMOIZED_ADAPTER, to_adapt_insp, equivalents=self.mapper._equivalent_columns, adapt_required=True, allow_label_resolve=False, anonymize_labels=True)
    assert clauses.is_aliased_class
    innerjoin = loadopt.local_opts.get('innerjoin', self.parent_property.innerjoin) if loadopt is not None else self.parent_property.innerjoin
    if not innerjoin:
        chained_from_outerjoin = True
    compile_state.create_eager_joins.append((self._create_eager_join, entity, path, adapter, parentmapper, clauses, innerjoin, chained_from_outerjoin, loadopt._extra_criteria if loadopt else ()))
    add_to_collection = compile_state.secondary_columns
    path.set(compile_state.attributes, 'eager_row_processor', clauses)
    return (clauses, adapter, add_to_collection, chained_from_outerjoin)