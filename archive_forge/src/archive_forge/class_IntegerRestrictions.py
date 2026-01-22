from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntegerRestrictions(_messages.Message):
    """Restrictions on INTEGER type values.

  Fields:
    maxValue: The maximum value that can be specified, if applicable.
    minValue: The minimum value that can be specified, if applicable.
  """
    maxValue = _messages.IntegerField(1)
    minValue = _messages.IntegerField(2)