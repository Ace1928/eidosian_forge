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
class ORMDMLState(AbstractORMCompileState):
    is_dml_returning = True
    from_statement_ctx: Optional[ORMFromStatementCompileState] = None

    @classmethod
    def _get_orm_crud_kv_pairs(cls, mapper, statement, kv_iterator, needs_to_be_cacheable):
        core_get_crud_kv_pairs = UpdateDMLState._get_crud_kv_pairs
        for k, v in kv_iterator:
            k = coercions.expect(roles.DMLColumnRole, k)
            if isinstance(k, str):
                desc = _entity_namespace_key(mapper, k, default=NO_VALUE)
                if desc is NO_VALUE:
                    yield (coercions.expect(roles.DMLColumnRole, k), coercions.expect(roles.ExpressionElementRole, v, type_=sqltypes.NullType(), is_crud=True) if needs_to_be_cacheable else v)
                else:
                    yield from core_get_crud_kv_pairs(statement, desc._bulk_update_tuples(v), needs_to_be_cacheable)
            elif 'entity_namespace' in k._annotations:
                k_anno = k._annotations
                attr = _entity_namespace_key(k_anno['entity_namespace'], k_anno['proxy_key'])
                yield from core_get_crud_kv_pairs(statement, attr._bulk_update_tuples(v), needs_to_be_cacheable)
            else:
                yield (k, v if not needs_to_be_cacheable else coercions.expect(roles.ExpressionElementRole, v, type_=sqltypes.NullType(), is_crud=True))

    @classmethod
    def _get_multi_crud_kv_pairs(cls, statement, kv_iterator):
        plugin_subject = statement._propagate_attrs['plugin_subject']
        if not plugin_subject or not plugin_subject.mapper:
            return UpdateDMLState._get_multi_crud_kv_pairs(statement, kv_iterator)
        return [dict(cls._get_orm_crud_kv_pairs(plugin_subject.mapper, statement, value_dict.items(), False)) for value_dict in kv_iterator]

    @classmethod
    def _get_crud_kv_pairs(cls, statement, kv_iterator, needs_to_be_cacheable):
        assert needs_to_be_cacheable, 'no test coverage for needs_to_be_cacheable=False'
        plugin_subject = statement._propagate_attrs['plugin_subject']
        if not plugin_subject or not plugin_subject.mapper:
            return UpdateDMLState._get_crud_kv_pairs(statement, kv_iterator, needs_to_be_cacheable)
        return list(cls._get_orm_crud_kv_pairs(plugin_subject.mapper, statement, kv_iterator, needs_to_be_cacheable))

    @classmethod
    def get_entity_description(cls, statement):
        ext_info = statement.table._annotations['parententity']
        mapper = ext_info.mapper
        if ext_info.is_aliased_class:
            _label_name = ext_info.name
        else:
            _label_name = mapper.class_.__name__
        return {'name': _label_name, 'type': mapper.class_, 'expr': ext_info.entity, 'entity': ext_info.entity, 'table': mapper.local_table}

    @classmethod
    def get_returning_column_descriptions(cls, statement):

        def _ent_for_col(c):
            return c._annotations.get('parententity', None)

        def _attr_for_col(c, ent):
            if ent is None:
                return c
            proxy_key = c._annotations.get('proxy_key', None)
            if not proxy_key:
                return c
            else:
                return getattr(ent.entity, proxy_key, c)
        return [{'name': c.key, 'type': c.type, 'expr': _attr_for_col(c, ent), 'aliased': ent.is_aliased_class, 'entity': ent.entity} for c, ent in [(c, _ent_for_col(c)) for c in statement._all_selected_columns]]

    def _setup_orm_returning(self, compiler, orm_level_statement, dml_level_statement, dml_mapper, *, use_supplemental_cols=True):
        """establish ORM column handlers for an INSERT, UPDATE, or DELETE
        which uses explicit returning().

        called within compilation level create_for_statement.

        The _return_orm_returning() method then receives the Result
        after the statement was executed, and applies ORM loading to the
        state that we first established here.

        """
        if orm_level_statement._returning:
            fs = FromStatement(orm_level_statement._returning, dml_level_statement, _adapt_on_names=False)
            fs = fs.execution_options(**orm_level_statement._execution_options)
            fs = fs.options(*orm_level_statement._with_options)
            self.select_statement = fs
            self.from_statement_ctx = fsc = ORMFromStatementCompileState.create_for_statement(fs, compiler)
            fsc.setup_dml_returning_compile_state(dml_mapper)
            dml_level_statement = dml_level_statement._generate()
            dml_level_statement._returning = ()
            cols_to_return = [c for c in fsc.primary_columns if c is not None]
            if not cols_to_return:
                cols_to_return.extend(dml_mapper.primary_key)
            if use_supplemental_cols:
                dml_level_statement = dml_level_statement.return_defaults(*dml_mapper.primary_key, supplemental_cols=cols_to_return)
            else:
                dml_level_statement = dml_level_statement.returning(*cols_to_return)
        return dml_level_statement

    @classmethod
    def _return_orm_returning(cls, session, statement, params, execution_options, bind_arguments, result):
        execution_context = result.context
        compile_state = execution_context.compiled.compile_state
        if compile_state.from_statement_ctx and (not compile_state.from_statement_ctx.compile_options._is_star):
            load_options = execution_options.get('_sa_orm_load_options', QueryContext.default_load_options)
            querycontext = QueryContext(compile_state.from_statement_ctx, compile_state.select_statement, params, session, load_options, execution_options, bind_arguments)
            return loading.instances(result, querycontext)
        else:
            return result