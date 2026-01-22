from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def _todb(table, dbo, tablename, schema=None, commit=True, truncate=False):
    if _is_clikchouse_dbapi_connection(dbo):
        debug('assuming %r is clickhosue DB-API 2.0 connection', dbo)
        _todb_clikchouse_dbapi_connection(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    elif _is_dbapi_connection(dbo):
        debug('assuming %r is standard DB-API 2.0 connection', dbo)
        _todb_dbapi_connection(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    elif _is_dbapi_cursor(dbo):
        debug('assuming %r is standard DB-API 2.0 cursor')
        _todb_dbapi_cursor(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    elif _is_sqlalchemy_engine(dbo):
        debug('assuming %r instance of sqlalchemy.engine.base.Engine', dbo)
        _todb_sqlalchemy_engine(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    elif _is_sqlalchemy_session(dbo):
        debug('assuming %r instance of sqlalchemy.orm.session.Session', dbo)
        _todb_sqlalchemy_session(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    elif _is_sqlalchemy_connection(dbo):
        debug('assuming %r instance of sqlalchemy.engine.base.Connection', dbo)
        _todb_sqlalchemy_connection(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    elif callable(dbo):
        debug('assuming %r is a function returning standard DB-API 2.0 cursor objects', dbo)
        _todb_dbapi_mkcurs(table, dbo, tablename, schema=schema, commit=commit, truncate=truncate)
    else:
        raise ArgumentError('unsupported database object type: %r' % dbo)