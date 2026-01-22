from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConditionContext(_messages.Message):
    """The IAM conditions context.

  Fields:
    accessTime: The hypothetical access timestamp to evaluate IAM conditions.
      Note that this value must not be earlier than the current time;
      otherwise, an INVALID_ARGUMENT error will be returned.
  """
    accessTime = _messages.StringField(1)