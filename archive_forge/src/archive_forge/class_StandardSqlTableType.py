from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardSqlTableType(_messages.Message):
    """A table type

  Fields:
    columns: The columns in this table type
  """
    columns = _messages.MessageField('StandardSqlField', 1, repeated=True)