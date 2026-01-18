import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def update_returning_pk(query, values, surrogate_key):
    """Perform an UPDATE, returning the primary key of the matched row.

    The primary key is returned using a selection of strategies:

    * if the database supports RETURNING, RETURNING is used to retrieve
      the primary key values inline.

    * If the database is MySQL and the entity is mapped to a single integer
      primary key column, MySQL's last_insert_id() function is used
      inline within the UPDATE and then upon a second SELECT to get the
      value.

    * Otherwise, a "refetch" strategy is used, where a given "surrogate"
      key value (typically a UUID column on the entity) is used to run
      a new SELECT against that UUID.   This UUID is also placed into
      the UPDATE query to ensure the row matches.

    :param query: a Query object with existing criterion, against a single
     entity.

    :param values: a dictionary of values to be updated on the row.

    :param surrogate_key: a tuple of (attrname, value), referring to a
     UNIQUE attribute that will also match the row.  This attribute is used
     to retrieve the row via a SELECT when no optimized strategy exists.

    :return: the primary key, returned as a tuple.
     Is only returned if rows matched is one.  Otherwise, CantUpdateException
     is raised.

    """
    entity = query.column_descriptions[0]['type']
    mapper = inspect(entity).mapper
    session = query.session
    bind = session.connection(bind_arguments=dict(mapper=mapper))
    if bind.dialect.name == 'postgresql':
        pk_strategy = _pk_strategy_returning
    elif bind.dialect.name == 'mysql' and len(mapper.primary_key) == 1 and isinstance(mapper.primary_key[0].type, sqltypes.Integer):
        pk_strategy = _pk_strategy_mysql_last_insert_id
    else:
        pk_strategy = _pk_strategy_refetch
    return pk_strategy(query, mapper, values, surrogate_key)