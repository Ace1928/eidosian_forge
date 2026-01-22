from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalProjectsApprovalRequestsInvalidateRequest(_messages.Message):
    """A AccessapprovalProjectsApprovalRequestsInvalidateRequest object.

  Fields:
    invalidateApprovalRequestMessage: A InvalidateApprovalRequestMessage
      resource to be passed as the request body.
    name: Name of the ApprovalRequest to invalidate.
  """
    invalidateApprovalRequestMessage = _messages.MessageField('InvalidateApprovalRequestMessage', 1)
    name = _messages.StringField(2, required=True)