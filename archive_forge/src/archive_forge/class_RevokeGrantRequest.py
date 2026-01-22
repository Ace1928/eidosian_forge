from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevokeGrantRequest(_messages.Message):
    """Request message for `RevokeGrant` method.

  Fields:
    reason: Optional. The reason for revoking this Grant.
  """
    reason = _messages.StringField(1)