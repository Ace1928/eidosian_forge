from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetTablePrimaryKey(_messages.Message):
    """Options to configure rule type SetTablePrimaryKey. The rule is used to
  specify the columns and name to configure/alter the primary key of a table.
  The rule filter field can refer to one entity. The rule scope can be one of:
  Table.

  Fields:
    primaryKey: Optional. Name for the primary key
    primaryKeyColumns: Required. List of column names for the primary key
  """
    primaryKey = _messages.StringField(1)
    primaryKeyColumns = _messages.StringField(2, repeated=True)