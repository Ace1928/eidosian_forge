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
@sql.base.CompileState.plugin_for('orm', 'orm_from_statement')
class ORMFromStatementCompileState(ORMCompileState):
    _from_obj_alias = None
    _has_mapper_entities = False
    statement_container: FromStatement
    requested_statement: Union[SelectBase, TextClause, UpdateBase]
    dml_table: Optional[_DMLTableElement] = None
    _has_orm_entities = False
    multi_row_eager_loaders = False
    eager_adding_joins = False
    compound_eager_adapter = None
    extra_criteria_entities = _EMPTY_DICT
    eager_joins = _EMPTY_DICT

    @classmethod
    def create_for_statement(cls, statement_container: Union[Select, FromStatement], compiler: Optional[SQLCompiler], **kw: Any) -> ORMFromStatementCompileState:
        assert isinstance(statement_container, FromStatement)
        if compiler is not None and compiler.stack:
            raise sa_exc.CompileError('The ORM FromStatement construct only supports being invoked as the topmost statement, as it is only intended to define how result rows should be returned.')
        self = cls.__new__(cls)
        self._primary_entity = None
        self.use_legacy_query_style = statement_container._compile_options._use_legacy_query_style
        self.statement_container = self.select_statement = statement_container
        self.requested_statement = statement = statement_container.element
        if statement.is_dml:
            self.dml_table = statement.table
            self.is_dml_returning = True
        self._entities = []
        self._polymorphic_adapters = {}
        self.compile_options = statement_container._compile_options
        if self.use_legacy_query_style and isinstance(statement, expression.SelectBase) and (not statement._is_textual) and (not statement.is_dml) and (statement._label_style is LABEL_STYLE_NONE):
            self.statement = statement.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        else:
            self.statement = statement
        self._label_convention = self._column_naming_convention(statement._label_style if not statement._is_textual and (not statement.is_dml) else LABEL_STYLE_NONE, self.use_legacy_query_style)
        _QueryEntity.to_compile_state(self, statement_container._raw_columns, self._entities, is_current_entities=True)
        self.current_path = statement_container._compile_options._current_path
        self._init_global_attributes(statement_container, compiler, process_criteria_for_toplevel=False, toplevel=True)
        if statement_container._with_options:
            for opt in statement_container._with_options:
                if opt._is_compile_state:
                    opt.process_compile_state(self)
        if statement_container._with_context_options:
            for fn, key in statement_container._with_context_options:
                fn(self)
        self.primary_columns = []
        self.secondary_columns = []
        self.dedupe_columns = set()
        self.create_eager_joins = []
        self._fallback_from_clauses = []
        self.order_by = None
        if isinstance(self.statement, expression.TextClause):
            self.extra_criteria_entities = {}
            for entity in self._entities:
                entity.setup_compile_state(self)
            compiler._ordered_columns = compiler._textual_ordered_columns = False
            compiler._loose_column_name_matching = True
            for c in self.primary_columns:
                compiler.process(c, within_columns_clause=True, add_to_result_map=compiler._add_to_result_map)
        else:
            self._from_obj_alias = ORMStatementAdapter(_TraceAdaptRole.ADAPT_FROM_STATEMENT, self.statement, adapt_on_names=statement_container._adapt_on_names)
        return self

    def _adapt_col_list(self, cols, current_adapter):
        return cols

    def _get_current_adapter(self):
        return None

    def setup_dml_returning_compile_state(self, dml_mapper):
        """used by BulkORMInsert (and Update / Delete?) to set up a handler
        for RETURNING to return ORM objects and expressions

        """
        target_mapper = self.statement._propagate_attrs.get('plugin_subject', None)
        adapter = DMLReturningColFilter(target_mapper, dml_mapper)
        if self.compile_options._is_star and len(self._entities) != 1:
            raise sa_exc.CompileError("Can't generate ORM query that includes multiple expressions at the same time as '*'; query for '*' alone if present")
        for entity in self._entities:
            entity.setup_dml_returning_compile_state(self, adapter)