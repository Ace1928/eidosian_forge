from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColumnFamily(_messages.Message):
    """A set of columns within a table which share a common configuration.

  Fields:
    gcRule: Garbage collection rule specified as a protobuf. Must serialize to
      at most 500 bytes. NOTE: Garbage collection executes opportunistically
      in the background, and so it's possible for reads to return a cell even
      if it matches the active GC expression for its family.
    stats: Output only. Only available with STATS_VIEW, this includes summary
      statistics about column family contents. For statistics over an entire
      table, see TableStats above.
    valueType: The type of data stored in each of this family's cell values,
      including its full encoding. If omitted, the family only serves raw
      untyped bytes. For now, only the `Aggregate` type is supported.
      `Aggregate` can only be set at family creation and is immutable
      afterwards. If `value_type` is `Aggregate`, written data must be
      compatible with: * `value_type.input_type` for `AddInput` mutations
  """
    gcRule = _messages.MessageField('GcRule', 1)
    stats = _messages.MessageField('ColumnFamilyStats', 2)
    valueType = _messages.MessageField('Type', 3)