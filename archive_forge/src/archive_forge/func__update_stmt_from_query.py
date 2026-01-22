import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def _update_stmt_from_query(mapper, query, values):
    upd_values = dict(((mapper.column_attrs[key], value) for key, value in values.items()))
    primary_table = inspect(query.column_descriptions[0]['entity']).local_table
    where_criteria = query.whereclause
    update_stmt = sql.update(primary_table).where(where_criteria).values(upd_values)
    return update_stmt