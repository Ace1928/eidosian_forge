from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
def _compound_eager_statement(self):
    if self.order_by:
        unwrapped_order_by = [elem.element if isinstance(elem, sql.elements._label_reference) else elem for elem in self.order_by]
        order_by_col_expr = sql_util.expand_column_list_from_order_by(self.primary_columns, unwrapped_order_by)
    else:
        order_by_col_expr = []
        unwrapped_order_by = None
    inner = self._select_statement(self.primary_columns + [c for c in order_by_col_expr if c not in self.dedupe_columns], self.from_clauses, self._where_criteria, self._having_criteria, self.label_style, self.order_by, for_update=self._for_update_arg, hints=self.select_statement._hints, statement_hints=self.select_statement._statement_hints, correlate=self.correlate, correlate_except=self.correlate_except, **self._select_args)
    inner = inner.alias()
    equivs = self._all_equivs()
    self.compound_eager_adapter = ORMStatementAdapter(_TraceAdaptRole.COMPOUND_EAGER_STATEMENT, inner, equivalents=equivs)
    statement = future.select(*[inner] + self.secondary_columns)
    statement._label_style = self.label_style
    if self._for_update_arg is not None and self._for_update_arg.of is None:
        statement._for_update_arg = self._for_update_arg
    from_clause = inner
    for eager_join in self.eager_joins.values():
        from_clause = sql_util.splice_joins(from_clause, eager_join, eager_join.stop_on)
    statement.select_from.non_generative(statement, from_clause)
    if unwrapped_order_by:
        statement.order_by.non_generative(statement, *self.compound_eager_adapter.copy_and_process(unwrapped_order_by))
    statement.order_by.non_generative(statement, *self.eager_order_by)
    return statement