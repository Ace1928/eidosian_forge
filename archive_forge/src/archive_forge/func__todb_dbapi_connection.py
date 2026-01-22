from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def _todb_dbapi_connection(table, connection, tablename, schema=None, commit=True, truncate=False):
    tablename = _quote(tablename)
    if schema is not None:
        tablename = _quote(schema) + '.' + tablename
    debug('tablename: %r', tablename)
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    colnames = [_quote(n) for n in flds]
    debug('column names: %r', colnames)
    placeholders = _placeholders(connection, colnames)
    debug('placeholders: %r', placeholders)
    cursor = connection.cursor()
    if truncate:
        truncatequery = SQL_TRUNCATE_QUERY % tablename
        debug('truncate the table via query %r', truncatequery)
        cursor.execute(truncatequery)
        cursor.close()
        cursor = connection.cursor()
    insertcolnames = ', '.join(colnames)
    insertquery = SQL_INSERT_QUERY % (tablename, insertcolnames, placeholders)
    debug('insert data via query %r' % insertquery)
    cursor.executemany(insertquery, it)
    debug('close the cursor')
    cursor.close()
    if commit:
        debug('commit transaction')
        connection.commit()