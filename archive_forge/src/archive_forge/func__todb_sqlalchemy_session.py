from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def _todb_sqlalchemy_session(table, session, tablename, schema=None, commit=True, truncate=False):
    _todb_sqlalchemy_connection(table, session.connection(), tablename, schema=schema, commit=commit, truncate=truncate)