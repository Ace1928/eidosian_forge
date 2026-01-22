from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CounterOptions(_messages.Message):
    """Options for counters

  Fields:
    field: The field value to attribute.
    metric: The metric to update.
  """
    field = _messages.StringField(1)
    metric = _messages.StringField(2)