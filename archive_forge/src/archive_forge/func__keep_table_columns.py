import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
def _keep_table_columns(self, schema, table_name, columns):
    assert table_name in schema.tables
    table = schema.tables[table_name]
    if not columns:
        return table
    new_columns = {}
    for column_name in columns:
        assert isinstance(column_name, str)
        assert column_name in table.columns
        new_columns[column_name] = table.columns[column_name]
    table.columns = new_columns
    return table