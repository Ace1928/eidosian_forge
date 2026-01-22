from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from collections import OrderedDict
import re
from apitools.base.py import extra_types
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import zip
@classmethod
def FromDdl(cls, database_ddl, table_name):
    """Constructs a Table from ddl statements.

    Args:
      database_ddl: String list, the ddl statements of the current table from
          server.
      table_name: String, the table name user inputs.

    Returns:
      Table.

    Raises:
      BadTableNameError: the table name is invalid.
      ValueError: Invalid Ddl.
    """
    table_name_list = []
    for ddl in database_ddl:
        table_match = cls._TABLE_DDL_PATTERN.search(ddl)
        if not table_match:
            continue
        name = table_match.group('name')
        if name != table_name:
            table_name_list.append(name)
            continue
        column_defs = table_match.group('columns')
        column_dict = OrderedDict()
        for column_ddl in column_defs.split(','):
            if column_ddl and (not column_ddl.isspace()):
                column = _TableColumn.FromDdl(column_ddl)
                column_dict[column.name] = column
        raw_primary_keys = table_match.groupdict()['primary_keys']
        primary_keys_list = [k.strip() for k in raw_primary_keys.split(',')]
        return Table(table_name, column_dict, primary_keys_list)
    raise BadTableNameError('Table name [{}] is invalid. Valid table names: [{}].'.format(table_name, ', '.join(table_name_list)))