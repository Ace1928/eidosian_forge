from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerSourceConfig(_messages.Message):
    """SQLServer data source configuration

  Fields:
    excludeObjects: SQLServer objects to exclude from the stream.
    includeObjects: SQLServer objects to include in the stream.
    maxConcurrentBackfillTasks: Max concurrent backfill tasks.
    maxConcurrentCdcTasks: Max concurrent CDC tasks.
  """
    excludeObjects = _messages.MessageField('SqlServerRdbms', 1)
    includeObjects = _messages.MessageField('SqlServerRdbms', 2)
    maxConcurrentBackfillTasks = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    maxConcurrentCdcTasks = _messages.IntegerField(4, variant=_messages.Variant.INT32)