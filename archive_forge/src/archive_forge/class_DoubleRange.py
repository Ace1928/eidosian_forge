from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DoubleRange(_messages.Message):
    """Range of a double hyperparameter.

  Fields:
    max: Max value of the double parameter.
    min: Min value of the double parameter.
  """
    max = _messages.FloatField(1)
    min = _messages.FloatField(2)