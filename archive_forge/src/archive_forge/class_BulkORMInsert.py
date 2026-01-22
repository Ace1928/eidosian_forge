from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@CompileState.plugin_for('orm', 'insert')
class BulkORMInsert(ORMDMLState, InsertDMLState):

    class default_insert_options(Options):
        _dml_strategy: DMLStrategyArgument = 'auto'
        _render_nulls: bool = False
        _return_defaults: bool = False
        _subject_mapper: Optional[Mapper[Any]] = None
        _autoflush: bool = True
        _populate_existing: bool = False
    select_statement: Optional[FromStatement] = None

    @classmethod
    def orm_pre_session_exec(cls, session, statement, params, execution_options, bind_arguments, is_pre_event):
        insert_options, execution_options = BulkORMInsert.default_insert_options.from_execution_options('_sa_orm_insert_options', {'dml_strategy', 'autoflush', 'populate_existing', 'render_nulls'}, execution_options, statement._execution_options)
        bind_arguments['clause'] = statement
        try:
            plugin_subject = statement._propagate_attrs['plugin_subject']
        except KeyError:
            assert False, "statement had 'orm' plugin but no plugin_subject"
        else:
            if plugin_subject:
                bind_arguments['mapper'] = plugin_subject.mapper
                insert_options += {'_subject_mapper': plugin_subject.mapper}
        if not params:
            if insert_options._dml_strategy == 'auto':
                insert_options += {'_dml_strategy': 'orm'}
            elif insert_options._dml_strategy == 'bulk':
                raise sa_exc.InvalidRequestError('Can\'t use "bulk" ORM insert strategy without passing separate parameters')
        elif insert_options._dml_strategy == 'auto':
            insert_options += {'_dml_strategy': 'bulk'}
        if insert_options._dml_strategy != 'raw':
            if not execution_options:
                execution_options = context._orm_load_exec_options
            else:
                execution_options = execution_options.union(context._orm_load_exec_options)
        if not is_pre_event and insert_options._autoflush:
            session._autoflush()
        statement = statement._annotate({'dml_strategy': insert_options._dml_strategy})
        return (statement, util.immutabledict(execution_options).union({'_sa_orm_insert_options': insert_options}))

    @classmethod
    def orm_execute_statement(cls, session: Session, statement: dml.Insert, params: _CoreAnyExecuteParams, execution_options: OrmExecuteOptionsParameter, bind_arguments: _BindArguments, conn: Connection) -> _result.Result:
        insert_options = execution_options.get('_sa_orm_insert_options', cls.default_insert_options)
        if insert_options._dml_strategy not in ('raw', 'bulk', 'orm', 'auto'):
            raise sa_exc.ArgumentError("Valid strategies for ORM insert strategy are 'raw', 'orm', 'bulk', 'auto")
        result: _result.Result[Any]
        if insert_options._dml_strategy == 'raw':
            result = conn.execute(statement, params or {}, execution_options=execution_options)
            return result
        if insert_options._dml_strategy == 'bulk':
            mapper = insert_options._subject_mapper
            if statement._post_values_clause is not None and mapper._multiple_persistence_tables:
                raise sa_exc.InvalidRequestError(f"bulk INSERT with a 'post values' clause (typically upsert) not supported for multi-table mapper {mapper}")
            assert mapper is not None
            assert session._transaction is not None
            result = _bulk_insert(mapper, cast('Iterable[Dict[str, Any]]', [params] if isinstance(params, dict) else params), session._transaction, isstates=False, return_defaults=insert_options._return_defaults, render_nulls=insert_options._render_nulls, use_orm_insert_stmt=statement, execution_options=execution_options)
        elif insert_options._dml_strategy == 'orm':
            result = conn.execute(statement, params or {}, execution_options=execution_options)
        else:
            raise AssertionError()
        if not bool(statement._returning):
            return result
        if insert_options._populate_existing:
            load_options = execution_options.get('_sa_orm_load_options', QueryContext.default_load_options)
            load_options += {'_populate_existing': True}
            execution_options = execution_options.union({'_sa_orm_load_options': load_options})
        return cls._return_orm_returning(session, statement, params, execution_options, bind_arguments, result)

    @classmethod
    def create_for_statement(cls, statement, compiler, **kw) -> BulkORMInsert:
        self = cast(BulkORMInsert, super().create_for_statement(statement, compiler, **kw))
        if compiler is not None:
            toplevel = not compiler.stack
        else:
            toplevel = True
        if not toplevel:
            return self
        mapper = statement._propagate_attrs['plugin_subject']
        dml_strategy = statement._annotations.get('dml_strategy', 'raw')
        if dml_strategy == 'bulk':
            self._setup_for_bulk_insert(compiler)
        elif dml_strategy == 'orm':
            self._setup_for_orm_insert(compiler, mapper)
        return self

    @classmethod
    def _resolved_keys_as_col_keys(cls, mapper, resolved_value_dict):
        return {col.key if col is not None else k: v for col, k, v in ((mapper.c.get(k), k, v) for k, v in resolved_value_dict.items())}

    def _setup_for_orm_insert(self, compiler, mapper):
        statement = orm_level_statement = cast(dml.Insert, self.statement)
        statement = self._setup_orm_returning(compiler, orm_level_statement, statement, dml_mapper=mapper, use_supplemental_cols=False)
        self.statement = statement

    def _setup_for_bulk_insert(self, compiler):
        """establish an INSERT statement within the context of
        bulk insert.

        This method will be within the "conn.execute()" call that is invoked
        by persistence._emit_insert_statement().

        """
        statement = orm_level_statement = cast(dml.Insert, self.statement)
        an = statement._annotations
        emit_insert_table, emit_insert_mapper = (an['_emit_insert_table'], an['_emit_insert_mapper'])
        statement = statement._clone()
        statement.table = emit_insert_table
        if self._dict_parameters:
            self._dict_parameters = {col: val for col, val in self._dict_parameters.items() if col.table is emit_insert_table}
        statement = self._setup_orm_returning(compiler, orm_level_statement, statement, dml_mapper=emit_insert_mapper, use_supplemental_cols=True)
        if self.from_statement_ctx is not None and self.from_statement_ctx.compile_options._is_star:
            raise sa_exc.CompileError("Can't use RETURNING * with bulk ORM INSERT.  Please use a different INSERT form, such as INSERT..VALUES or INSERT with a Core Connection")
        self.statement = statement