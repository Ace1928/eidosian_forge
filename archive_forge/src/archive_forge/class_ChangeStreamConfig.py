from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChangeStreamConfig(_messages.Message):
    """Change stream configuration.

  Fields:
    retentionPeriod: How long the change stream should be retained. Change
      stream data older than the retention period will not be returned when
      reading the change stream from the table. Values must be at least 1 day
      and at most 7 days, and will be truncated to microsecond granularity.
  """
    retentionPeriod = _messages.StringField(1)