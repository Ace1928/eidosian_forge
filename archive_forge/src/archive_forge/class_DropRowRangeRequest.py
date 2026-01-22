from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DropRowRangeRequest(_messages.Message):
    """Request message for
  google.bigtable.admin.v2.BigtableTableAdmin.DropRowRange

  Fields:
    deleteAllDataFromTable: Delete all rows in the table. Setting this to
      false is a no-op.
    rowKeyPrefix: Delete all rows that start with this row key prefix. Prefix
      cannot be zero length.
  """
    deleteAllDataFromTable = _messages.BooleanField(1)
    rowKeyPrefix = _messages.BytesField(2)