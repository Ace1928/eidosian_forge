from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IsInvitableUserResponse(_messages.Message):
    """Response for IsInvitableUser RPC.

  Fields:
    isInvitableUser: Returns true if the email address is invitable.
  """
    isInvitableUser = _messages.BooleanField(1)