from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleSourceConfig(_messages.Message):
    """Oracle data source configuration

  Fields:
    dropLargeObjects: Drop large object values.
    excludeObjects: Oracle objects to exclude from the stream.
    includeObjects: Oracle objects to include in the stream.
    maxConcurrentBackfillTasks: Maximum number of concurrent backfill tasks.
      The number should be non-negative. If not set (or set to 0), the
      system's default value is used.
    maxConcurrentCdcTasks: Maximum number of concurrent CDC tasks. The number
      should be non-negative. If not set (or set to 0), the system's default
      value is used.
    streamLargeObjects: Stream large object values.
  """
    dropLargeObjects = _messages.MessageField('DropLargeObjects', 1)
    excludeObjects = _messages.MessageField('OracleRdbms', 2)
    includeObjects = _messages.MessageField('OracleRdbms', 3)
    maxConcurrentBackfillTasks = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    maxConcurrentCdcTasks = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    streamLargeObjects = _messages.MessageField('StreamLargeObjects', 6)