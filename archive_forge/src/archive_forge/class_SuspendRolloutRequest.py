from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SuspendRolloutRequest(_messages.Message):
    """Message for suspending a rollout.

  Fields:
    reason: Optional. Reason for suspension.
  """
    reason = _messages.StringField(1)