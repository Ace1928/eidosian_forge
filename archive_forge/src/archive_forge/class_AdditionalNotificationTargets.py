from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdditionalNotificationTargets(_messages.Message):
    """AdditionalNotificationTargets includes email addresses to be notified.

  Fields:
    adminEmailRecipients: Optional. Additional email addresses to be notified
      when a principal(requester) is granted access.
    requesterEmailRecipients: Optional. Additional email address to be
      notified about an eligible entitlement.
  """
    adminEmailRecipients = _messages.StringField(1, repeated=True)
    requesterEmailRecipients = _messages.StringField(2, repeated=True)