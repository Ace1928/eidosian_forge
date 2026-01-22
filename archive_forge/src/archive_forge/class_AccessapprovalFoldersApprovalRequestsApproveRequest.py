from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalFoldersApprovalRequestsApproveRequest(_messages.Message):
    """A AccessapprovalFoldersApprovalRequestsApproveRequest object.

  Fields:
    approveApprovalRequestMessage: A ApproveApprovalRequestMessage resource to
      be passed as the request body.
    name: Name of the approval request to approve.
  """
    approveApprovalRequestMessage = _messages.MessageField('ApproveApprovalRequestMessage', 1)
    name = _messages.StringField(2, required=True)