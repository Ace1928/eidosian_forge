from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DailyCycle(_messages.Message):
    """Time window specified for daily operations.

  Fields:
    duration: Output only. Duration of the time window, set by service
      producer.
    startTime: Time within the day to start the operations.
  """
    duration = _messages.StringField(1)
    startTime = _messages.MessageField('TimeOfDay', 2)