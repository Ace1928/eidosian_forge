from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryRestriction(_messages.Message):
    """Specifies query restrictions to apply. This allows UI to provide common
  filter needs (e.g. timestamps) without having the user to write them in SQL.

  Fields:
    timerange: Optional. This restriction is the TIME_RANGE restriction type
      in the QueryRestrictionConflict. Range is [start_time, end_time).
      Granularity: down to milliseconds (not nanoseconds)
  """
    timerange = _messages.MessageField('Interval', 1)