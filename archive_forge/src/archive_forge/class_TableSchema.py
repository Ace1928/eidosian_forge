from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableSchema(_messages.Message):
    """A TableSchema object.

  Fields:
    fields: Describes the fields in a table.
  """
    fields = _messages.MessageField('TableFieldSchema', 1, repeated=True)