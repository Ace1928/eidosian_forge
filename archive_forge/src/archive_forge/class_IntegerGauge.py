from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntegerGauge(_messages.Message):
    """A metric value representing temporal values of a variable.

  Fields:
    timestamp: The time at which this value was measured. Measured as msecs
      from epoch.
    value: The value of the variable represented by this gauge.
  """
    timestamp = _messages.StringField(1)
    value = _messages.MessageField('SplitInt64', 2)