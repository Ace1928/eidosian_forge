import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
def _filter_schema(schema, schema_tables, exclude_table_columns):
    """Filters a schema to only include the specified tables in the
       schema_tables parameter.  This will also filter out any colums for
       included tables that reference tables that are not included
       in the schema_tables parameter

    :param schema: Schema dict to be filtered
    :param schema_tables: List of table names to filter on.
                          EX: ['Bridge', 'Controller', 'Interface']
                          NOTE: This list is case sensitive.
    :return: Schema dict:
                filtered if the schema_table parameter contains table names,
                else the original schema dict
    """
    tables = {}
    for tbl_name, tbl_data in schema['tables'].items():
        if not schema_tables or tbl_name in schema_tables:
            columns = {}
            exclude_columns = exclude_table_columns.get(tbl_name, [])
            for col_name, col_data in tbl_data['columns'].items():
                if col_name in exclude_columns:
                    continue
                type_ = col_data.get('type')
                if type_:
                    if type_ and isinstance(type_, dict):
                        key = type_.get('key')
                        if key and isinstance(key, dict):
                            ref_tbl = key.get('refTable')
                            if ref_tbl and isinstance(ref_tbl, str):
                                if ref_tbl not in schema_tables:
                                    continue
                        value = type_.get('value')
                        if value and isinstance(value, dict):
                            ref_tbl = value.get('refTable')
                            if ref_tbl and isinstance(ref_tbl, str):
                                if ref_tbl not in schema_tables:
                                    continue
                columns[col_name] = col_data
            tbl_data['columns'] = columns
            tables[tbl_name] = tbl_data
    schema['tables'] = tables
    return schema