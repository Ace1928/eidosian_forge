from collections import namedtuple
import functools
import hashlib
import re
from peewee import *
from peewee import CommaNodeList
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import callable_
from peewee import sort_models
from peewee import sqlite3
from peewee import _truncate_constraint_name
class SqliteMigrator(SchemaMigrator):
    """
    SQLite supports a subset of ALTER TABLE queries, view the docs for the
    full details http://sqlite.org/lang_altertable.html
    """
    column_re = re.compile('(.+?)\\((.+)\\)')
    column_split_re = re.compile('(?:[^,(]|\\([^)]*\\))+')
    column_name_re = re.compile('["`\']?([\\w]+)')
    fk_re = re.compile('FOREIGN KEY\\s+\\("?([\\w]+)"?\\)\\s+', re.I)

    def _get_column_names(self, table):
        res = self.database.execute_sql('select * from "%s" limit 1' % table)
        return [item[0] for item in res.description]

    def _get_create_table(self, table):
        res = self.database.execute_sql('select name, sql from sqlite_master where type=? and LOWER(name)=?', ['table', table.lower()])
        return res.fetchone()

    @operation
    def _update_column(self, table, column_to_update, fn):
        columns = set((column.name.lower() for column in self.database.get_columns(table)))
        if column_to_update.lower() not in columns:
            raise ValueError('Column "%s" does not exist on "%s"' % (column_to_update, table))
        table, create_table = self._get_create_table(table)
        indexes = self.database.get_indexes(table)
        self.database.get_foreign_keys(table)
        create_table = re.sub('\\s+', ' ', create_table)
        raw_create, raw_columns = self.column_re.search(create_table).groups()
        split_columns = self.column_split_re.findall(raw_columns)
        column_defs = [col.strip() for col in split_columns]
        new_column_defs = []
        new_column_names = []
        original_column_names = []
        constraint_terms = ('foreign ', 'primary ', 'constraint ', 'check ')
        for column_def in column_defs:
            column_name, = self.column_name_re.match(column_def).groups()
            if column_name == column_to_update:
                new_column_def = fn(column_name, column_def)
                if new_column_def:
                    new_column_defs.append(new_column_def)
                    original_column_names.append(column_name)
                    column_name, = self.column_name_re.match(new_column_def).groups()
                    new_column_names.append(column_name)
            else:
                new_column_defs.append(column_def)
                if not column_def.lower().startswith(constraint_terms):
                    new_column_names.append(column_name)
                    original_column_names.append(column_name)
        original_to_new = dict(zip(original_column_names, new_column_names))
        new_column = original_to_new.get(column_to_update)
        fk_filter_fn = lambda column_def: column_def
        if not new_column:
            fk_filter_fn = lambda column_def: None
        elif new_column != column_to_update:
            fk_filter_fn = lambda column_def: self.fk_re.sub('FOREIGN KEY ("%s") ' % new_column, column_def)
        cleaned_columns = []
        for column_def in new_column_defs:
            match = self.fk_re.match(column_def)
            if match is not None and match.groups()[0] == column_to_update:
                column_def = fk_filter_fn(column_def)
            if column_def:
                cleaned_columns.append(column_def)
        temp_table = table + '__tmp__'
        rgx = re.compile('("?)%s("?)' % table, re.I)
        create = rgx.sub('\\1%s\\2' % temp_table, raw_create)
        columns = ', '.join(cleaned_columns)
        queries = [NodeList([SQL('DROP TABLE IF EXISTS'), Entity(temp_table)]), SQL('%s (%s)' % (create.strip(), columns))]
        populate_table = NodeList((SQL('INSERT INTO'), Entity(temp_table), EnclosedNodeList([Entity(col) for col in new_column_names]), SQL('SELECT'), CommaNodeList([Entity(col) for col in original_column_names]), SQL('FROM'), Entity(table)))
        drop_original = NodeList([SQL('DROP TABLE'), Entity(table)])
        queries += [populate_table, drop_original, self.rename_table(temp_table, table)]
        for index in filter(lambda idx: idx.sql, indexes):
            if column_to_update not in index.columns:
                queries.append(SQL(index.sql))
            elif new_column:
                sql = self._fix_index(index.sql, column_to_update, new_column)
                if sql is not None:
                    queries.append(SQL(sql))
        return queries

    def _fix_index(self, sql, column_to_update, new_column):
        parts = sql.split(column_to_update)
        if len(parts) == 2:
            return sql.replace(column_to_update, new_column)
        lhs, rhs = sql.rsplit('(', 1)
        if len(rhs.split(column_to_update)) == 2:
            return '%s(%s' % (lhs, rhs.replace(column_to_update, new_column))
        parts = rhs.rsplit(')', 1)[0].split(',')
        columns = [part.strip('"`[]\' ') for part in parts]
        clean = []
        for column in columns:
            if re.match('%s(?:[\\\'"`\\]]?\\s|$)' % column_to_update, column):
                column = new_column + column[len(column_to_update):]
            clean.append(column)
        return '%s(%s)' % (lhs, ', '.join(('"%s"' % c for c in clean)))

    @operation
    def drop_column(self, table, column_name, cascade=True, legacy=False):
        if sqlite3.sqlite_version_info >= (3, 35, 0) and (not legacy):
            ctx = self.make_context()
            self._alter_table(ctx, table).literal(' DROP COLUMN ').sql(Entity(column_name))
            return ctx
        return self._update_column(table, column_name, lambda a, b: None)

    @operation
    def rename_column(self, table, old_name, new_name, legacy=False):
        if sqlite3.sqlite_version_info >= (3, 25, 0) and (not legacy):
            return self._alter_table(self.make_context(), table).literal(' RENAME COLUMN ').sql(Entity(old_name)).literal(' TO ').sql(Entity(new_name))

        def _rename(column_name, column_def):
            return column_def.replace(column_name, new_name)
        return self._update_column(table, old_name, _rename)

    @operation
    def add_not_null(self, table, column):

        def _add_not_null(column_name, column_def):
            return column_def + ' NOT NULL'
        return self._update_column(table, column, _add_not_null)

    @operation
    def drop_not_null(self, table, column):

        def _drop_not_null(column_name, column_def):
            return column_def.replace('NOT NULL', '')
        return self._update_column(table, column, _drop_not_null)

    @operation
    def add_column_default(self, table, column, default):
        if default is None:
            raise ValueError('`default` must be not None/NULL.')
        if callable_(default):
            default = default()
        if isinstance(default, str) and (not default.endswith((')', "'"))) and (not default.isdigit()):
            default = "'%s'" % default

        def _add_default(column_name, column_def):
            return column_def + ' DEFAULT %s' % default
        return self._update_column(table, column, _add_default)

    @operation
    def drop_column_default(self, table, column):

        def _drop_default(column_name, column_def):
            col = re.sub('DEFAULT\\s+[\\w"\\\'\\(\\)]+(\\s|$)', '', column_def, re.I)
            return col.strip()
        return self._update_column(table, column, _drop_default)

    @operation
    def alter_column_type(self, table, column, field, cast=None):
        if cast is not None:
            raise ValueError('alter_column_type() does not support cast with Sqlite.')
        ctx = self.make_context()

        def _alter_column_type(column_name, column_def):
            node_list = field.ddl(ctx)
            sql, _ = ctx.sql(Entity(column)).sql(node_list).query()
            return sql
        return self._update_column(table, column, _alter_column_type)

    @operation
    def add_constraint(self, table, name, constraint):
        raise NotImplementedError

    @operation
    def drop_constraint(self, table, name):
        raise NotImplementedError

    @operation
    def add_foreign_key_constraint(self, table, column_name, field, on_delete=None, on_update=None):
        raise NotImplementedError