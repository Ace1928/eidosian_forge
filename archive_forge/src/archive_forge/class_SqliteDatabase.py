from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class SqliteDatabase(Database):
    field_types = {'BIGAUTO': FIELD.AUTO, 'BIGINT': FIELD.INT, 'BOOL': FIELD.INT, 'DOUBLE': FIELD.FLOAT, 'SMALLINT': FIELD.INT, 'UUID': FIELD.TEXT}
    operations = {'LIKE': 'GLOB', 'ILIKE': 'LIKE'}
    index_schema_prefix = True
    limit_max = -1
    server_version = __sqlite_version__
    truncate_table = False

    def __init__(self, database, *args, **kwargs):
        self._pragmas = kwargs.pop('pragmas', ())
        super(SqliteDatabase, self).__init__(database, *args, **kwargs)
        self._aggregates = {}
        self._collations = {}
        self._functions = {}
        self._window_functions = {}
        self._table_functions = []
        self._extensions = set()
        self._attached = {}
        self.register_function(_sqlite_date_part, 'date_part', 2)
        self.register_function(_sqlite_date_trunc, 'date_trunc', 2)
        self.nulls_ordering = self.server_version >= (3, 30, 0)

    def init(self, database, pragmas=None, timeout=5, returning_clause=None, **kwargs):
        if pragmas is not None:
            self._pragmas = pragmas
        if isinstance(self._pragmas, dict):
            self._pragmas = list(self._pragmas.items())
        if returning_clause is not None:
            if __sqlite_version__ < (3, 35, 0):
                warnings.warn('RETURNING clause requires Sqlite 3.35 or newer')
            self.returning_clause = returning_clause
        self._timeout = timeout
        super(SqliteDatabase, self).init(database, **kwargs)

    def _set_server_version(self, conn):
        pass

    def _connect(self):
        if sqlite3 is None:
            raise ImproperlyConfigured('SQLite driver not installed!')
        conn = sqlite3.connect(self.database, timeout=self._timeout, isolation_level=None, **self.connect_params)
        try:
            self._add_conn_hooks(conn)
        except:
            conn.close()
            raise
        return conn

    def _add_conn_hooks(self, conn):
        if self._attached:
            self._attach_databases(conn)
        if self._pragmas:
            self._set_pragmas(conn)
        self._load_aggregates(conn)
        self._load_collations(conn)
        self._load_functions(conn)
        if self.server_version >= (3, 25, 0):
            self._load_window_functions(conn)
        if self._table_functions:
            for table_function in self._table_functions:
                table_function.register(conn)
        if self._extensions:
            self._load_extensions(conn)

    def _set_pragmas(self, conn):
        cursor = conn.cursor()
        for pragma, value in self._pragmas:
            cursor.execute('PRAGMA %s = %s;' % (pragma, value))
        cursor.close()

    def _attach_databases(self, conn):
        cursor = conn.cursor()
        for name, db in self._attached.items():
            cursor.execute('ATTACH DATABASE "%s" AS "%s"' % (db, name))
        cursor.close()

    def pragma(self, key, value=SENTINEL, permanent=False, schema=None):
        if schema is not None:
            key = '"%s".%s' % (schema, key)
        sql = 'PRAGMA %s' % key
        if value is not SENTINEL:
            sql += ' = %s' % (value or 0)
            if permanent:
                pragmas = dict(self._pragmas or ())
                pragmas[key] = value
                self._pragmas = list(pragmas.items())
        elif permanent:
            raise ValueError('Cannot specify a permanent pragma without value')
        row = self.execute_sql(sql).fetchone()
        if row:
            return row[0]
    cache_size = __pragma__('cache_size')
    foreign_keys = __pragma__('foreign_keys')
    journal_mode = __pragma__('journal_mode')
    journal_size_limit = __pragma__('journal_size_limit')
    mmap_size = __pragma__('mmap_size')
    page_size = __pragma__('page_size')
    read_uncommitted = __pragma__('read_uncommitted')
    synchronous = __pragma__('synchronous')
    wal_autocheckpoint = __pragma__('wal_autocheckpoint')
    application_id = __pragma__('application_id')
    user_version = __pragma__('user_version')
    data_version = __pragma__('data_version')

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, seconds):
        if self._timeout == seconds:
            return
        self._timeout = seconds
        if not self.is_closed():
            self.execute_sql('PRAGMA busy_timeout=%d;' % (seconds * 1000))

    def _load_aggregates(self, conn):
        for name, (klass, num_params) in self._aggregates.items():
            conn.create_aggregate(name, num_params, klass)

    def _load_collations(self, conn):
        for name, fn in self._collations.items():
            conn.create_collation(name, fn)

    def _load_functions(self, conn):
        for name, (fn, n_params, deterministic) in self._functions.items():
            kwargs = {'deterministic': deterministic} if deterministic else {}
            conn.create_function(name, n_params, fn, **kwargs)

    def _load_window_functions(self, conn):
        for name, (klass, num_params) in self._window_functions.items():
            conn.create_window_function(name, num_params, klass)

    def register_aggregate(self, klass, name=None, num_params=-1):
        self._aggregates[name or klass.__name__.lower()] = (klass, num_params)
        if not self.is_closed():
            self._load_aggregates(self.connection())

    def aggregate(self, name=None, num_params=-1):

        def decorator(klass):
            self.register_aggregate(klass, name, num_params)
            return klass
        return decorator

    def register_collation(self, fn, name=None):
        name = name or fn.__name__

        def _collation(*args):
            expressions = args + (SQL('collate %s' % name),)
            return NodeList(expressions)
        fn.collation = _collation
        self._collations[name] = fn
        if not self.is_closed():
            self._load_collations(self.connection())

    def collation(self, name=None):

        def decorator(fn):
            self.register_collation(fn, name)
            return fn
        return decorator

    def register_function(self, fn, name=None, num_params=-1, deterministic=None):
        self._functions[name or fn.__name__] = (fn, num_params, deterministic)
        if not self.is_closed():
            self._load_functions(self.connection())

    def func(self, name=None, num_params=-1, deterministic=None):

        def decorator(fn):
            self.register_function(fn, name, num_params, deterministic)
            return fn
        return decorator

    def register_window_function(self, klass, name=None, num_params=-1):
        name = name or klass.__name__.lower()
        self._window_functions[name] = (klass, num_params)
        if not self.is_closed():
            self._load_window_functions(self.connection())

    def window_function(self, name=None, num_params=-1):

        def decorator(klass):
            self.register_window_function(klass, name, num_params)
            return klass
        return decorator

    def register_table_function(self, klass, name=None):
        if name is not None:
            klass.name = name
        self._table_functions.append(klass)
        if not self.is_closed():
            klass.register(self.connection())

    def table_function(self, name=None):

        def decorator(klass):
            self.register_table_function(klass, name)
            return klass
        return decorator

    def unregister_aggregate(self, name):
        del self._aggregates[name]

    def unregister_collation(self, name):
        del self._collations[name]

    def unregister_function(self, name):
        del self._functions[name]

    def unregister_window_function(self, name):
        del self._window_functions[name]

    def unregister_table_function(self, name):
        for idx, klass in enumerate(self._table_functions):
            if klass.name == name:
                break
        else:
            return False
        self._table_functions.pop(idx)
        return True

    def _load_extensions(self, conn):
        conn.enable_load_extension(True)
        for extension in self._extensions:
            conn.load_extension(extension)

    def load_extension(self, extension):
        self._extensions.add(extension)
        if not self.is_closed():
            conn = self.connection()
            conn.enable_load_extension(True)
            conn.load_extension(extension)

    def unload_extension(self, extension):
        self._extensions.remove(extension)

    def attach(self, filename, name):
        if name in self._attached:
            if self._attached[name] == filename:
                return False
            raise OperationalError('schema "%s" already attached.' % name)
        self._attached[name] = filename
        if not self.is_closed():
            self.execute_sql('ATTACH DATABASE "%s" AS "%s"' % (filename, name))
        return True

    def detach(self, name):
        if name not in self._attached:
            return False
        del self._attached[name]
        if not self.is_closed():
            self.execute_sql('DETACH DATABASE "%s"' % name)
        return True

    def last_insert_id(self, cursor, query_type=None):
        if not self.returning_clause:
            return cursor.lastrowid
        elif query_type == Insert.SIMPLE:
            try:
                return cursor[0][0]
            except (IndexError, KeyError, TypeError):
                pass
        return cursor

    def rows_affected(self, cursor):
        try:
            return cursor.rowcount
        except AttributeError:
            return cursor.cursor.rowcount

    def begin(self, lock_type=None):
        statement = 'BEGIN %s' % lock_type if lock_type else 'BEGIN'
        self.execute_sql(statement)

    def commit(self):
        with __exception_wrapper__:
            return self._state.conn.commit()

    def rollback(self):
        with __exception_wrapper__:
            return self._state.conn.rollback()

    def get_tables(self, schema=None):
        schema = schema or 'main'
        cursor = self.execute_sql('SELECT name FROM "%s".sqlite_master WHERE type=? ORDER BY name' % schema, ('table',))
        return [row for row, in cursor.fetchall()]

    def get_views(self, schema=None):
        sql = 'SELECT name, sql FROM "%s".sqlite_master WHERE type=? ORDER BY name' % (schema or 'main')
        return [ViewMetadata(*row) for row in self.execute_sql(sql, ('view',))]

    def get_indexes(self, table, schema=None):
        schema = schema or 'main'
        query = 'SELECT name, sql FROM "%s".sqlite_master WHERE tbl_name = ? AND type = ? ORDER BY name' % schema
        cursor = self.execute_sql(query, (table, 'index'))
        index_to_sql = dict(cursor.fetchall())
        unique_indexes = set()
        cursor = self.execute_sql('PRAGMA "%s".index_list("%s")' % (schema, table))
        for row in cursor.fetchall():
            name = row[1]
            is_unique = int(row[2]) == 1
            if is_unique:
                unique_indexes.add(name)
        index_columns = {}
        for index_name in sorted(index_to_sql):
            cursor = self.execute_sql('PRAGMA "%s".index_info("%s")' % (schema, index_name))
            index_columns[index_name] = [row[2] for row in cursor.fetchall()]
        return [IndexMetadata(name, index_to_sql[name], index_columns[name], name in unique_indexes, table) for name in sorted(index_to_sql)]

    def get_columns(self, table, schema=None):
        cursor = self.execute_sql('PRAGMA "%s".table_info("%s")' % (schema or 'main', table))
        return [ColumnMetadata(r[1], r[2], not r[3], bool(r[5]), table, r[4]) for r in cursor.fetchall()]

    def get_primary_keys(self, table, schema=None):
        cursor = self.execute_sql('PRAGMA "%s".table_info("%s")' % (schema or 'main', table))
        return [row[1] for row in filter(lambda r: r[-1], cursor.fetchall())]

    def get_foreign_keys(self, table, schema=None):
        cursor = self.execute_sql('PRAGMA "%s".foreign_key_list("%s")' % (schema or 'main', table))
        return [ForeignKeyMetadata(row[3], row[2], row[4], table) for row in cursor.fetchall()]

    def get_binary_type(self):
        return sqlite3.Binary

    def conflict_statement(self, on_conflict, query):
        action = on_conflict._action.lower() if on_conflict._action else ''
        if action and action not in ('nothing', 'update'):
            return SQL('INSERT OR %s' % on_conflict._action.upper())

    def conflict_update(self, oc, query):
        if self.server_version < (3, 24, 0) and any((oc._preserve, oc._update, oc._where, oc._conflict_target, oc._conflict_constraint)):
            raise ValueError('SQLite does not support specifying which values to preserve or update.')
        action = oc._action.lower() if oc._action else ''
        if action and action not in ('nothing', 'update', ''):
            return
        if action == 'nothing':
            return SQL('ON CONFLICT DO NOTHING')
        elif not oc._update and (not oc._preserve):
            raise ValueError('If you are not performing any updates (or preserving any INSERTed values), then the conflict resolution action should be set to "NOTHING".')
        elif oc._conflict_constraint:
            raise ValueError('SQLite does not support specifying named constraints for conflict resolution.')
        elif not oc._conflict_target:
            raise ValueError('SQLite requires that a conflict target be specified when doing an upsert.')
        return self._build_on_conflict_update(oc, query)

    def extract_date(self, date_part, date_field):
        return fn.date_part(date_part, date_field, python_value=int)

    def truncate_date(self, date_part, date_field):
        return fn.date_trunc(date_part, date_field, python_value=simple_date_time)

    def to_timestamp(self, date_field):
        return fn.strftime('%s', date_field).cast('integer')

    def from_timestamp(self, date_field):
        return fn.datetime(date_field, 'unixepoch')