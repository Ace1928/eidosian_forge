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
class _ORMColumnEntity(_ColumnEntity):
    """Column/expression based entity."""
    supports_single_entity = False
    __slots__ = ('expr', 'mapper', 'column', '_label_name', 'entity_zero_or_selectable', 'entity_zero', '_extra_entities')

    def __init__(self, compile_state, column, entities_collection, parententity, raw_column_index, is_current_entities, parent_bundle=None):
        annotations = column._annotations
        _entity = parententity
        orm_key = annotations.get('proxy_key', None)
        proxy_owner = annotations.get('proxy_owner', _entity)
        if orm_key:
            self.expr = getattr(proxy_owner.entity, orm_key)
            self.translate_raw_column = False
        else:
            self.expr = column
            self.translate_raw_column = raw_column_index is not None
        self.raw_column_index = raw_column_index
        if is_current_entities:
            self._label_name = compile_state._label_convention(column, col_name=orm_key)
        else:
            self._label_name = None
        _entity._post_inspect
        self.entity_zero = self.entity_zero_or_selectable = ezero = _entity
        self.mapper = mapper = _entity.mapper
        if parent_bundle:
            parent_bundle._entities.append(self)
        else:
            entities_collection.append(self)
        compile_state._has_orm_entities = True
        self.column = column
        self._fetch_column = self._row_processor = None
        self._extra_entities = (self.expr, self.column)
        if mapper._should_select_with_poly_adapter:
            compile_state._create_with_polymorphic_adapter(ezero, ezero.selectable)

    def corresponds_to(self, entity):
        if _is_aliased_class(entity):
            return entity is self.entity_zero
        else:
            return not _is_aliased_class(self.entity_zero) and entity.common_parent(self.entity_zero)

    def setup_dml_returning_compile_state(self, compile_state: ORMCompileState, adapter: DMLReturningColFilter) -> None:
        self._fetch_column = self.column
        column = adapter(self.column, False)
        if column is not None:
            compile_state.dedupe_columns.add(column)
            compile_state.primary_columns.append(column)

    def setup_compile_state(self, compile_state):
        current_adapter = compile_state._get_current_adapter()
        if current_adapter:
            column = current_adapter(self.column, False)
            if column is None:
                assert compile_state.is_dml_returning
                self._fetch_column = self.column
                return
        else:
            column = self.column
        ezero = self.entity_zero
        single_table_crit = self.mapper._single_table_criterion
        if single_table_crit is not None or ('additional_entity_criteria', self.mapper) in compile_state.global_attributes:
            compile_state.extra_criteria_entities[ezero] = (ezero, ezero._adapter if ezero.is_aliased_class else None)
        if column._annotations and (not column._expression_label):
            column = column._deannotate()
        if set(self.column._from_objects).intersection(ezero.selectable._from_objects):
            compile_state._fallback_from_clauses.append(ezero.selectable)
        compile_state.dedupe_columns.add(column)
        compile_state.primary_columns.append(column)
        self._fetch_column = column