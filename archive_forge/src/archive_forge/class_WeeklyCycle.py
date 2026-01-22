from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WeeklyCycle(_messages.Message):
    """Time window specified for weekly operations.

  Fields:
    schedule: User can specify multiple windows in a week. Minimum of 1
      window.
  """
    schedule = _messages.MessageField('Schedule', 1, repeated=True)