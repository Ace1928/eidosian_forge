from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SetProjectOptOutStateRequest(_messages.Message):
    """Request message for KmsOptOutService.SetProjectOptOutState.

  Fields:
    value: Required. New opt out preference value (true == opt out, and vice
      versa).
  """
    value = _messages.BooleanField(1)