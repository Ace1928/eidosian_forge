import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def _pk_strategy_returning(query, mapper, values, surrogate_key):
    surrogate_key_name, surrogate_key_value = surrogate_key
    surrogate_key_col = mapper.attrs[surrogate_key_name].expression
    update_stmt = _update_stmt_from_query(mapper, query, values)
    update_stmt = update_stmt.where(surrogate_key_col == surrogate_key_value)
    update_stmt = update_stmt.returning(*mapper.primary_key)
    result = query.session.execute(update_stmt)
    rowcount = result.rowcount
    _assert_single_row(rowcount)
    primary_key = tuple(result.first())
    return primary_key