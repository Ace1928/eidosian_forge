from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
class PostgresqlMetadata(Metadata):
    column_map = {16: BooleanField, 17: BlobField, 20: BigIntegerField, 21: SmallIntegerField, 23: IntegerField, 25: TextField, 700: FloatField, 701: DoubleField, 1042: CharField, 1043: CharField, 1082: DateField, 1114: DateTimeField, 1184: DateTimeField, 1083: TimeField, 1266: TimeField, 1700: DecimalField, 2950: UUIDField}
    array_types = {1000: BooleanField, 1001: BlobField, 1005: SmallIntegerField, 1007: IntegerField, 1009: TextField, 1014: CharField, 1015: CharField, 1016: BigIntegerField, 1115: DateTimeField, 1182: DateField, 1183: TimeField, 2951: UUIDField}
    extension_import = 'from playhouse.postgres_ext import *'

    def __init__(self, database):
        super(PostgresqlMetadata, self).__init__(database)
        if postgres_ext is not None:
            cursor = self.execute('select oid, typname, format_type(oid, NULL) from pg_type;')
            results = cursor.fetchall()
            for oid, typname, formatted_type in results:
                if typname == 'json':
                    self.column_map[oid] = postgres_ext.JSONField
                elif typname == 'jsonb':
                    self.column_map[oid] = postgres_ext.BinaryJSONField
                elif typname == 'hstore':
                    self.column_map[oid] = postgres_ext.HStoreField
                elif typname == 'tsvector':
                    self.column_map[oid] = postgres_ext.TSVectorField
            for oid in self.array_types:
                self.column_map[oid] = postgres_ext.ArrayField

    def get_column_types(self, table, schema):
        column_types = {}
        extra_params = {}
        extension_types = set((postgres_ext.ArrayField, postgres_ext.BinaryJSONField, postgres_ext.JSONField, postgres_ext.TSVectorField, postgres_ext.HStoreField)) if postgres_ext is not None else set()
        identifier = '%s."%s"' % (schema, table)
        cursor = self.execute('SELECT attname, atttypid FROM pg_catalog.pg_attribute WHERE attrelid = %s::regclass AND attnum > %s', identifier, 0)
        for name, oid in cursor.fetchall():
            column_types[name] = self.column_map.get(oid, UnknownField)
            if column_types[name] in extension_types:
                self.requires_extension = True
            if oid in self.array_types:
                extra_params[name] = {'field_class': self.array_types[oid]}
        return (column_types, extra_params)

    def get_columns(self, table, schema=None):
        schema = schema or 'public'
        return super(PostgresqlMetadata, self).get_columns(table, schema)

    def get_foreign_keys(self, table, schema=None):
        schema = schema or 'public'
        return super(PostgresqlMetadata, self).get_foreign_keys(table, schema)

    def get_primary_keys(self, table, schema=None):
        schema = schema or 'public'
        return super(PostgresqlMetadata, self).get_primary_keys(table, schema)

    def get_indexes(self, table, schema=None):
        schema = schema or 'public'
        return super(PostgresqlMetadata, self).get_indexes(table, schema)