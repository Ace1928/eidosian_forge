from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MysqlSourceConfig(_messages.Message):
    """MySQL source configuration

  Fields:
    excludeObjects: MySQL objects to exclude from the stream.
    includeObjects: MySQL objects to retrieve from the source.
    maxConcurrentBackfillTasks: Maximum number of concurrent backfill tasks.
      The number should be non negative. If not set (or set to 0), the
      system's default value will be used.
    maxConcurrentCdcTasks: Maximum number of concurrent CDC tasks. The number
      should be non negative. If not set (or set to 0), the system's default
      value will be used.
  """
    excludeObjects = _messages.MessageField('MysqlRdbms', 1)
    includeObjects = _messages.MessageField('MysqlRdbms', 2)
    maxConcurrentBackfillTasks = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    maxConcurrentCdcTasks = _messages.IntegerField(4, variant=_messages.Variant.INT32)