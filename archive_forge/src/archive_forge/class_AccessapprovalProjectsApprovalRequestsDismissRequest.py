from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalProjectsApprovalRequestsDismissRequest(_messages.Message):
    """A AccessapprovalProjectsApprovalRequestsDismissRequest object.

  Fields:
    dismissApprovalRequestMessage: A DismissApprovalRequestMessage resource to
      be passed as the request body.
    name: Name of the ApprovalRequest to dismiss.
  """
    dismissApprovalRequestMessage = _messages.MessageField('DismissApprovalRequestMessage', 1)
    name = _messages.StringField(2, required=True)