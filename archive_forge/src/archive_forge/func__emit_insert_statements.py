from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def _emit_insert_statements(base_mapper, uowtransaction, mapper, table, insert, *, bookkeeping=True, use_orm_insert_stmt=None, execution_options=None):
    """Emit INSERT statements corresponding to value lists collected
    by _collect_insert_commands()."""
    if use_orm_insert_stmt is not None:
        cached_stmt = use_orm_insert_stmt
        exec_opt = util.EMPTY_DICT
        returning_is_required_anyway = bool(use_orm_insert_stmt._returning)
        deterministic_results_reqd = returning_is_required_anyway and use_orm_insert_stmt._sort_by_parameter_order or bookkeeping
    else:
        returning_is_required_anyway = False
        deterministic_results_reqd = bookkeeping
        cached_stmt = base_mapper._memo(('insert', table), table.insert)
        exec_opt = {'compiled_cache': base_mapper._compiled_cache}
    if execution_options:
        execution_options = util.EMPTY_DICT.merge_with(exec_opt, execution_options)
    else:
        execution_options = exec_opt
    return_result = None
    for (connection, _, hasvalue, has_all_pks, has_all_defaults), records in groupby(insert, lambda rec: (rec[4], set(rec[2]), bool(rec[5]), rec[6], rec[7])):
        statement = cached_stmt
        if use_orm_insert_stmt is not None:
            statement = statement._annotate({'_emit_insert_table': table, '_emit_insert_mapper': mapper})
        if (not bookkeeping or (has_all_defaults or not base_mapper._prefer_eager_defaults(connection.dialect, table) or (not table.implicit_returning) or (not connection.dialect.insert_returning))) and (not returning_is_required_anyway) and has_all_pks and (not hasvalue):
            records = list(records)
            multiparams = [rec[2] for rec in records]
            result = connection.execute(statement, multiparams, execution_options=execution_options)
            if bookkeeping:
                for (state, state_dict, params, mapper_rec, conn, value_params, has_all_pks, has_all_defaults), last_inserted_params in zip(records, result.context.compiled_parameters):
                    if state:
                        _postfetch(mapper_rec, uowtransaction, table, state, state_dict, result, last_inserted_params, value_params, False, result.returned_defaults if not result.context.executemany else None)
                    else:
                        _postfetch_bulk_save(mapper_rec, state_dict, table)
        else:
            records = list(records)
            if returning_is_required_anyway or (table.implicit_returning and (not hasvalue) and (len(records) > 1)):
                if deterministic_results_reqd and connection.dialect.insert_executemany_returning_sort_by_parameter_order or (not deterministic_results_reqd and connection.dialect.insert_executemany_returning):
                    do_executemany = True
                elif returning_is_required_anyway:
                    if deterministic_results_reqd:
                        dt = ' with RETURNING and sort by parameter order'
                    else:
                        dt = ' with RETURNING'
                    raise sa_exc.InvalidRequestError(f"Can't use explicit RETURNING for bulk INSERT operation with {connection.dialect.dialect_description} backend; executemany{dt} is not enabled for this dialect.")
                else:
                    do_executemany = False
            else:
                do_executemany = False
            if use_orm_insert_stmt is None:
                if not has_all_defaults and base_mapper._prefer_eager_defaults(connection.dialect, table):
                    statement = statement.return_defaults(*mapper._server_default_cols[table], sort_by_parameter_order=bookkeeping)
            if mapper.version_id_col is not None:
                statement = statement.return_defaults(mapper.version_id_col, sort_by_parameter_order=bookkeeping)
            elif do_executemany:
                statement = statement.return_defaults(*table.primary_key, sort_by_parameter_order=bookkeeping)
            if do_executemany:
                multiparams = [rec[2] for rec in records]
                result = connection.execute(statement, multiparams, execution_options=execution_options)
                if use_orm_insert_stmt is not None:
                    if return_result is None:
                        return_result = result
                    else:
                        return_result = return_result.splice_vertically(result)
                if bookkeeping:
                    for (state, state_dict, params, mapper_rec, conn, value_params, has_all_pks, has_all_defaults), last_inserted_params, inserted_primary_key, returned_defaults in zip_longest(records, result.context.compiled_parameters, result.inserted_primary_key_rows, result.returned_defaults_rows or ()):
                        if inserted_primary_key is None:
                            raise orm_exc.FlushError('Multi-row INSERT statement for %s did not produce the correct number of INSERTed rows for RETURNING.  Ensure there are no triggers or special driver issues preventing INSERT from functioning properly.' % mapper_rec)
                        for pk, col in zip(inserted_primary_key, mapper._pks_by_table[table]):
                            prop = mapper_rec._columntoproperty[col]
                            if state_dict.get(prop.key) is None:
                                state_dict[prop.key] = pk
                        if state:
                            _postfetch(mapper_rec, uowtransaction, table, state, state_dict, result, last_inserted_params, value_params, False, returned_defaults)
                        else:
                            _postfetch_bulk_save(mapper_rec, state_dict, table)
            else:
                assert not returning_is_required_anyway
                for state, state_dict, params, mapper_rec, connection, value_params, has_all_pks, has_all_defaults in records:
                    if value_params:
                        result = connection.execute(statement.values(value_params), params, execution_options=execution_options)
                    else:
                        result = connection.execute(statement, params, execution_options=execution_options)
                    primary_key = result.inserted_primary_key
                    if primary_key is None:
                        raise orm_exc.FlushError('Single-row INSERT statement for %s did not produce a new primary key result being invoked.  Ensure there are no triggers or special driver issues preventing INSERT from functioning properly.' % (mapper_rec,))
                    for pk, col in zip(primary_key, mapper._pks_by_table[table]):
                        prop = mapper_rec._columntoproperty[col]
                        if col in value_params or state_dict.get(prop.key) is None:
                            state_dict[prop.key] = pk
                    if bookkeeping:
                        if state:
                            _postfetch(mapper_rec, uowtransaction, table, state, state_dict, result, result.context.compiled_parameters[0], value_params, False, result.returned_defaults if not result.context.executemany else None)
                        else:
                            _postfetch_bulk_save(mapper_rec, state_dict, table)
    if use_orm_insert_stmt is not None:
        if return_result is None:
            return _cursor.null_dml_result()
        else:
            return return_result