from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GetProjectOptOutStateResponse(_messages.Message):
    """Response message for KmsOptOutService.GetProjectOptOutState.

  Fields:
    value: The current opt-out preference (true == opt out, and vice versa).
  """
    value = _messages.BooleanField(1)