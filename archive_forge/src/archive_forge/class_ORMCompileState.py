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
class ORMCompileState(AbstractORMCompileState):

    class default_compile_options(CacheableOptions):
        _cache_key_traversal = [('_use_legacy_query_style', InternalTraversal.dp_boolean), ('_for_statement', InternalTraversal.dp_boolean), ('_bake_ok', InternalTraversal.dp_boolean), ('_current_path', InternalTraversal.dp_has_cache_key), ('_enable_single_crit', InternalTraversal.dp_boolean), ('_enable_eagerloads', InternalTraversal.dp_boolean), ('_only_load_props', InternalTraversal.dp_plain_obj), ('_set_base_alias', InternalTraversal.dp_boolean), ('_for_refresh_state', InternalTraversal.dp_boolean), ('_render_for_subquery', InternalTraversal.dp_boolean), ('_is_star', InternalTraversal.dp_boolean)]
        _use_legacy_query_style = False
        _for_statement = False
        _bake_ok = True
        _current_path = _path_registry
        _enable_single_crit = True
        _enable_eagerloads = True
        _only_load_props = None
        _set_base_alias = False
        _for_refresh_state = False
        _render_for_subquery = False
        _is_star = False
    attributes: Dict[Any, Any]
    global_attributes: Dict[Any, Any]
    statement: Union[Select[Any], FromStatement[Any]]
    select_statement: Union[Select[Any], FromStatement[Any]]
    _entities: List[_QueryEntity]
    _polymorphic_adapters: Dict[_InternalEntityType, ORMAdapter]
    compile_options: Union[Type[default_compile_options], default_compile_options]
    _primary_entity: Optional[_QueryEntity]
    use_legacy_query_style: bool
    _label_convention: _LabelConventionCallable
    primary_columns: List[ColumnElement[Any]]
    secondary_columns: List[ColumnElement[Any]]
    dedupe_columns: Set[ColumnElement[Any]]
    create_eager_joins: List[Tuple[Any, ...]]
    current_path: PathRegistry = _path_registry
    _has_mapper_entities = False

    def __init__(self, *arg, **kw):
        raise NotImplementedError()
    if TYPE_CHECKING:

        @classmethod
        def create_for_statement(cls, statement: Union[Select, FromStatement], compiler: Optional[SQLCompiler], **kw: Any) -> ORMCompileState:
            ...

    def _append_dedupe_col_collection(self, obj, col_collection):
        dedupe = self.dedupe_columns
        if obj not in dedupe:
            dedupe.add(obj)
            col_collection.append(obj)

    @classmethod
    def _column_naming_convention(cls, label_style: SelectLabelStyle, legacy: bool) -> _LabelConventionCallable:
        if legacy:

            def name(col, col_name=None):
                if col_name:
                    return col_name
                else:
                    return getattr(col, 'key')
            return name
        else:
            return SelectState._column_naming_convention(label_style)

    @classmethod
    def get_column_descriptions(cls, statement):
        return _column_descriptions(statement)

    @classmethod
    def orm_pre_session_exec(cls, session, statement, params, execution_options, bind_arguments, is_pre_event):
        load_options, execution_options = QueryContext.default_load_options.from_execution_options('_sa_orm_load_options', {'populate_existing', 'autoflush', 'yield_per', 'identity_token', 'sa_top_level_orm_context'}, execution_options, statement._execution_options)
        if 'sa_top_level_orm_context' in execution_options:
            ctx = execution_options['sa_top_level_orm_context']
            execution_options = ctx.query._execution_options.merge_with(ctx.execution_options, execution_options)
        if not execution_options:
            execution_options = _orm_load_exec_options
        else:
            execution_options = execution_options.union(_orm_load_exec_options)
        if load_options._yield_per:
            execution_options = execution_options.union({'yield_per': load_options._yield_per})
        if getattr(statement._compile_options, '_current_path', None) and len(statement._compile_options._current_path) > 10 and (execution_options.get('compiled_cache', True) is not None):
            execution_options: util.immutabledict[str, Any] = execution_options.union({'compiled_cache': None, '_cache_disable_reason': 'excess depth for ORM loader options'})
        bind_arguments['clause'] = statement
        try:
            plugin_subject = statement._propagate_attrs['plugin_subject']
        except KeyError:
            assert False, "statement had 'orm' plugin but no plugin_subject"
        else:
            if plugin_subject:
                bind_arguments['mapper'] = plugin_subject.mapper
        if not is_pre_event and load_options._autoflush:
            session._autoflush()
        return (statement, execution_options)

    @classmethod
    def orm_setup_cursor_result(cls, session, statement, params, execution_options, bind_arguments, result):
        execution_context = result.context
        compile_state = execution_context.compiled.compile_state
        load_options = execution_options.get('_sa_orm_load_options', QueryContext.default_load_options)
        if compile_state.compile_options._is_star:
            return result
        querycontext = QueryContext(compile_state, statement, params, session, load_options, execution_options, bind_arguments)
        return loading.instances(result, querycontext)

    @property
    def _lead_mapper_entities(self):
        """return all _MapperEntity objects in the lead entities collection.

        Does **not** include entities that have been replaced by
        with_entities(), with_only_columns()

        """
        return [ent for ent in self._entities if isinstance(ent, _MapperEntity)]

    def _create_with_polymorphic_adapter(self, ext_info, selectable):
        """given MapperEntity or ORMColumnEntity, setup polymorphic loading
        if called for by the Mapper.

        As of #8168 in 2.0.0rc1, polymorphic adapters, which greatly increase
        the complexity of the query creation process, are not used at all
        except in the quasi-legacy cases of with_polymorphic referring to an
        alias and/or subquery. This would apply to concrete polymorphic
        loading, and joined inheritance where a subquery is
        passed to with_polymorphic (which is completely unnecessary in modern
        use).

        """
        if not ext_info.is_aliased_class and ext_info.mapper.persist_selectable not in self._polymorphic_adapters:
            for mp in ext_info.mapper.iterate_to_root():
                self._mapper_loads_polymorphically_with(mp, ORMAdapter(_TraceAdaptRole.WITH_POLYMORPHIC_ADAPTER, mp, equivalents=mp._equivalent_columns, selectable=selectable))

    def _mapper_loads_polymorphically_with(self, mapper, adapter):
        for m2 in mapper._with_polymorphic_mappers or [mapper]:
            self._polymorphic_adapters[m2] = adapter
            for m in m2.iterate_to_root():
                self._polymorphic_adapters[m.local_table] = adapter

    @classmethod
    def _create_entities_collection(cls, query, legacy):
        raise NotImplementedError('this method only works for ORMSelectCompileState')