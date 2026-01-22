import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
class DbSchema(object):
    """Schema for an OVSDB database."""

    def __init__(self, name, version, tables, allow_extensions=False):
        self.name = name
        self.version = version
        self.tables = tables
        if self.__root_set_size() == 0:
            for table in self.tables.values():
                table.is_root = True
        for table in self.tables.values():
            for column in table.columns.values():
                self.__follow_ref_table(column, column.type.key, 'key')
                self.__follow_ref_table(column, column.type.value, 'value')

    def __root_set_size(self):
        """Returns the number of tables in the schema's root set."""
        n_root = 0
        for table in self.tables.values():
            if table.is_root:
                n_root += 1
        return n_root

    @staticmethod
    def from_json(json, allow_extensions=False):
        parser = ovs.db.parser.Parser(json, 'database schema')
        name = parser.get('name', ['id'])
        version = parser.get_optional('version', (str,))
        parser.get_optional('cksum', (str,))
        tablesJson = parser.get('tables', [dict])
        parser.finish()
        if version is not None and (not re.match('[0-9]+\\.[0-9]+\\.[0-9]+$', version)):
            raise error.Error('schema version "%s" not in format x.y.z' % version)
        tables = {}
        for tableName, tableJson in tablesJson.items():
            _check_id(tableName, json)
            tables[tableName] = TableSchema.from_json(tableJson, tableName, allow_extensions)
        return DbSchema(name, version, tables)

    def to_json(self):
        default_is_root = self.__root_set_size() == len(self.tables)
        tables = {}
        for table in self.tables.values():
            tables[table.name] = table.to_json(default_is_root)
        json = {'name': self.name, 'tables': tables}
        if self.version:
            json['version'] = self.version
        return json

    def copy(self):
        return DbSchema.from_json(self.to_json())

    def __follow_ref_table(self, column, base, base_name):
        if not base or base.type != ovs.db.types.UuidType or (not base.ref_table_name):
            return
        base.ref_table = self.tables.get(base.ref_table_name)
        if not base.ref_table:
            raise error.Error('column %s %s refers to undefined table %s' % (column.name, base_name, base.ref_table_name), tag='syntax error')
        if base.is_strong_ref() and (not base.ref_table.is_root):
            column.persistent = True