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
class SchemaMigrator(object):
    explicit_create_foreign_key = False
    explicit_delete_foreign_key = False

    def __init__(self, database):
        self.database = database

    def make_context(self):
        return self.database.get_sql_context()

    @classmethod
    def from_database(cls, database):
        if CockroachDatabase and isinstance(database, CockroachDatabase):
            return CockroachDBMigrator(database)
        elif isinstance(database, PostgresqlDatabase):
            return PostgresqlMigrator(database)
        elif isinstance(database, MySQLDatabase):
            return MySQLMigrator(database)
        elif isinstance(database, SqliteDatabase):
            return SqliteMigrator(database)
        raise ValueError('Unsupported database: %s' % database)

    @operation
    def apply_default(self, table, column_name, field):
        default = field.default
        if callable_(default):
            default = default()
        return self.make_context().literal('UPDATE ').sql(Entity(table)).literal(' SET ').sql(Expression(Entity(column_name), OP.EQ, field.db_value(default), flat=True))

    def _alter_table(self, ctx, table):
        return ctx.literal('ALTER TABLE ').sql(Entity(table))

    def _alter_column(self, ctx, table, column):
        return self._alter_table(ctx, table).literal(' ALTER COLUMN ').sql(Entity(column))

    @operation
    def alter_add_column(self, table, column_name, field):
        ctx = self.make_context()
        field_null, field.null = (field.null, True)
        if field.column_name != column_name:
            field.name = field.column_name = column_name
        self._alter_table(ctx, table).literal(' ADD COLUMN ').sql(field.ddl(ctx))
        field.null = field_null
        if isinstance(field, ForeignKeyField):
            self.add_inline_fk_sql(ctx, field)
        return ctx

    @operation
    def add_constraint(self, table, name, constraint):
        return self._alter_table(self.make_context(), table).literal(' ADD CONSTRAINT ').sql(Entity(name)).literal(' ').sql(constraint)

    @operation
    def add_unique(self, table, *column_names):
        constraint_name = 'uniq_%s' % '_'.join(column_names)
        constraint = NodeList((SQL('UNIQUE'), EnclosedNodeList([Entity(column) for column in column_names])))
        return self.add_constraint(table, constraint_name, constraint)

    @operation
    def drop_constraint(self, table, name):
        return self._alter_table(self.make_context(), table).literal(' DROP CONSTRAINT ').sql(Entity(name))

    def add_inline_fk_sql(self, ctx, field):
        ctx = ctx.literal(' REFERENCES ').sql(Entity(field.rel_model._meta.table_name)).literal(' ').sql(EnclosedNodeList((Entity(field.rel_field.column_name),)))
        if field.on_delete is not None:
            ctx = ctx.literal(' ON DELETE %s' % field.on_delete)
        if field.on_update is not None:
            ctx = ctx.literal(' ON UPDATE %s' % field.on_update)
        return ctx

    @operation
    def add_foreign_key_constraint(self, table, column_name, rel, rel_column, on_delete=None, on_update=None):
        constraint = 'fk_%s_%s_refs_%s' % (table, column_name, rel)
        ctx = self.make_context().literal('ALTER TABLE ').sql(Entity(table)).literal(' ADD CONSTRAINT ').sql(Entity(_truncate_constraint_name(constraint))).literal(' FOREIGN KEY ').sql(EnclosedNodeList((Entity(column_name),))).literal(' REFERENCES ').sql(Entity(rel)).literal(' (').sql(Entity(rel_column)).literal(')')
        if on_delete is not None:
            ctx = ctx.literal(' ON DELETE %s' % on_delete)
        if on_update is not None:
            ctx = ctx.literal(' ON UPDATE %s' % on_update)
        return ctx

    @operation
    def add_column(self, table, column_name, field):
        if not field.null and field.default is None:
            raise ValueError('%s is not null but has no default' % column_name)
        is_foreign_key = isinstance(field, ForeignKeyField)
        if is_foreign_key and (not field.rel_field):
            raise ValueError('Foreign keys must specify a `field`.')
        operations = [self.alter_add_column(table, column_name, field)]
        if not field.null:
            operations.extend([self.apply_default(table, column_name, field), self.add_not_null(table, column_name)])
        if is_foreign_key and self.explicit_create_foreign_key:
            operations.append(self.add_foreign_key_constraint(table, column_name, field.rel_model._meta.table_name, field.rel_field.column_name, field.on_delete, field.on_update))
        if field.index or field.unique:
            using = getattr(field, 'index_type', None)
            operations.append(self.add_index(table, (column_name,), field.unique, using))
        return operations

    @operation
    def drop_foreign_key_constraint(self, table, column_name):
        raise NotImplementedError

    @operation
    def drop_column(self, table, column_name, cascade=True):
        ctx = self.make_context()
        self._alter_table(ctx, table).literal(' DROP COLUMN ').sql(Entity(column_name))
        if cascade:
            ctx.literal(' CASCADE')
        fk_columns = [foreign_key.column for foreign_key in self.database.get_foreign_keys(table)]
        if column_name in fk_columns and self.explicit_delete_foreign_key:
            return [self.drop_foreign_key_constraint(table, column_name), ctx]
        return ctx

    @operation
    def rename_column(self, table, old_name, new_name):
        return self._alter_table(self.make_context(), table).literal(' RENAME COLUMN ').sql(Entity(old_name)).literal(' TO ').sql(Entity(new_name))

    @operation
    def add_not_null(self, table, column):
        return self._alter_column(self.make_context(), table, column).literal(' SET NOT NULL')

    @operation
    def drop_not_null(self, table, column):
        return self._alter_column(self.make_context(), table, column).literal(' DROP NOT NULL')

    @operation
    def add_column_default(self, table, column, default):
        if default is None:
            raise ValueError('`default` must be not None/NULL.')
        if callable_(default):
            default = default()
        if isinstance(default, str) and default.endswith((')', "'")):
            default = SQL(default)
        return self._alter_table(self.make_context(), table).literal(' ALTER COLUMN ').sql(Entity(column)).literal(' SET DEFAULT ').sql(default)

    @operation
    def drop_column_default(self, table, column):
        return self._alter_table(self.make_context(), table).literal(' ALTER COLUMN ').sql(Entity(column)).literal(' DROP DEFAULT')

    @operation
    def alter_column_type(self, table, column, field, cast=None):
        ctx = self.make_context()
        ctx = self._alter_column(ctx, table, column).literal(' TYPE ').sql(field.ddl_datatype(ctx))
        if cast is not None:
            if not isinstance(cast, Node):
                cast = SQL(cast)
            ctx = ctx.literal(' USING ').sql(cast)
        return ctx

    @operation
    def rename_table(self, old_name, new_name):
        return self._alter_table(self.make_context(), old_name).literal(' RENAME TO ').sql(Entity(new_name))

    @operation
    def add_index(self, table, columns, unique=False, using=None):
        ctx = self.make_context()
        index_name = make_index_name(table, columns)
        table_obj = Table(table)
        cols = [getattr(table_obj.c, column) for column in columns]
        index = Index(index_name, table_obj, cols, unique=unique, using=using)
        return ctx.sql(index)

    @operation
    def drop_index(self, table, index_name):
        return self.make_context().literal('DROP INDEX ').sql(Entity(index_name))