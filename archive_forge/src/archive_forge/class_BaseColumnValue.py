from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class BaseColumnValue(_messages.Message):
    """Optional identifier of the base column. If present, this column is
    derived from the specified base column.

    Fields:
      columnId: The id of the column in the base table from which this column
        is derived.
      tableIndex: Offset to the entry in the list of base tables in the table
        definition.
    """
    columnId = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    tableIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)